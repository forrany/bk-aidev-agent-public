import json
import re
import uuid
from importlib.metadata import version as pkg_version
from logging import getLogger
from typing import Any, Callable, ClassVar, Generator, List, Optional

from ag_ui.core import BaseEvent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.stores import ByteStore
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from aidev_agent.config import settings
from aidev_agent.core.ag_ui.aidev_agent import AidevAGUIAgent
from aidev_agent.core.ag_ui.types import AgentInput
from aidev_agent.core.ag_ui.utils import langchain_messages_to_agui
from aidev_agent.core.tools.runtime_tools import RuntimeBackendResolver
from aidev_agent.enums import AgentType, PromptRole
from aidev_agent.exceptions import AgentException
from aidev_agent.packages.langchain_core.models.llm_gateway import ChatModel
from aidev_agent.pydantic_models import (
    AgentOptions,
    ChatPrompt,
    ExecuteKwargs,
    KnowledgeSettings,
    ModelContextSettings,
)
from aidev_agent.services.agent.registry import AgentBuildContext, ChatBuildExtras
from aidev_agent.services.common_agent import CommonAgentProtocol, CommonQAAgent
from aidev_agent.services.event_handlers.base import BaseSessionWriter
from aidev_agent.services.messages_handler import GeneratorStreamingHelper
from aidev_agent.utils.async_utils import async_to_sync_generator
from aidev_agent.utils.loop import run_coro_sync
from aidev_agent.utils.migrations import (
    migration_chat_model_non_thinking_from_non_thinking_llm_v1,
    migration_knowledge_query_options_from_agent_options_v1,
    migration_model_context_options_from_agent_options_v1,
)

logger = getLogger(__name__)


class ChatCompletionAgent(BaseModel):
    """聊天 Agent"""

    agent_type: ClassVar[AgentType] = AgentType.CHAT

    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chat_model: BaseChatModel | None = None
    """聊天模型；种子实例（``ChatCompletionAgent()``）为 ``None``，``build(ctx)``
    装配后由 :meth:`ChatAgentBuilder.build_chat_model` 填充为非空 ``BaseChatModel``。
    种子实例不可执行（``execute()`` 假设非空）。"""
    chat_model_non_thinking: BaseChatModel | None = None
    """非思考模型；由 :meth:`ChatAgentBuilder.build_chat_model_non_thinking` 填充。"""
    non_thinking_llm: str | None = Field(default=None, deprecated="Use chat_model_non_thinking instead")
    chat_history: list[ChatPrompt] | None = None
    files: list[dict] = Field(default_factory=list)
    tools: Optional[list[StructuredTool]] = None
    skills: Optional[list] = None
    executor_info: Optional[dict] = None
    knowledge_bases: Optional[list[dict]] = None
    knowledges: Optional[list[dict]] = None
    knowledge_query_options: Any = None
    model_context_options: ModelContextSettings | None = None
    agent_options: AgentOptions | None = Field(
        default=None,
        deprecated="Use model_context_options and knowledge_query_options instead",
    )
    support_vision: bool = False
    file_store: ByteStore | None = None
    max_token_size: int | None = None
    callbacks: list[BaseCallbackHandler] | None = None
    agent_cls: CommonAgentProtocol = Field(default_factory=CommonQAAgent)
    """通用 agent 实例（实现 ``CommonAgentProtocol``）；ChatCompletionAgent 在 ``_get_agent`` 阶段
    通过 ``self.agent_cls.get_agent_executor(...)`` 触发执行器构建。
    字段名保留为 ``agent_cls`` 避免外部破坏，但语义已是「实例」。"""
    messages: list[BaseMessage] = Field(default_factory=list)
    checkpointer: BaseCheckpointSaver | None = None

    event_handler: Callable[[BaseEvent], None] | None = None
    mcp_fetch_failures: list[dict] = Field(default_factory=list, description="MCP 工具拉取失败记录，用于流式事件")
    resource_manager: Any = Field(
        default=None, exclude=True, description="per-request 资源管理器（含正确 app_code / access_token）"
    )
    runtime_backend_resolver: Any = Field(
        default=None,
        exclude=True,
        description="RuntimeBackendResolver 引用，由 ChatAgentBuilder 构造，用于执行结束后关闭沙箱资源",
    )
    agent_info: dict | None = Field(default=None, description="原始配置信息，来自 AgentConfig.agent_info")

    IMAGE_FILE_PATTERN: ClassVar[re.Pattern] = re.compile(r"^!\[.*\]\((http[^)]+/([^/]+?))\)")
    TOOL_EXECUTION_INTERVAL: ClassVar[int] = 10
    UPLOAD_IMAGE_PROMPT_PREFIX: ClassVar[Any] = "我上传了个图片文件,文件名为{file_name}。"
    SKIP_PROMPT_ROLE: ClassVar[list[str]] = ["guide", "reasoning"]

    class Config:
        arbitrary_types_allowed = True

    # ---------- 公共入口 ----------

    def build(self, ctx: AgentBuildContext) -> "ChatCompletionAgent":
        """在 ``self``（``cls()`` 空种子实例）上原地装配 fully-built ``ChatCompletionAgent`` 并返回。

        Chat 专属装配逻辑由 :class:`ChatAgentBuilder` 承接，本方法仅做字段赋值。
        所有装配参数来自 ``ctx``（通用字段）和 ``ctx.chat``（Chat 专属字段），
        不再依赖 factory 反向引用。

        可选字段（``agent_cls`` / ``thread_id`` / ``event_handler``）仅在 ctx 提供
        非空值时覆盖，缺省时保留种子默认值（如 ``thread_id`` 的随机 uuid）。
        """
        chat = ctx.chat or ChatBuildExtras()
        builder = ChatAgentBuilder(ctx)
        builder.handle_agent_switch()

        self.chat_model = builder.build_chat_model()
        self.chat_model_non_thinking = builder.build_chat_model_non_thinking()
        self.skills = builder.build_skills()
        # 先构建 executor_info，供 build_tools / construct_mcp 使用同一凭证源
        self.executor_info = builder.build_executor_info()
        self.resource_manager = ctx.resource_manager
        self.tools = builder.build_tools()
        self.mcp_fetch_failures = builder.mcp_fetch_failures
        self.knowledge_bases = builder.build_knowledge_bases()
        self.knowledges = builder.build_knowledge_items()
        self.knowledge_query_options = builder.build_knowledge_query_options()
        self.model_context_options = builder.build_model_context_options()
        if ctx.agent_config and ctx.agent_config.agent_options is not None:
            self.agent_options = ctx.agent_config.agent_options
        self.support_vision = builder.build_support_vision()
        self.chat_history = builder.build_chat_history(ctx.session_context_data)
        self.checkpointer = builder.build_checkpointer()
        self.callbacks = chat.callbacks
        # 构造 RuntimeBackendResolver（在 ChatAgentBuilder 层管理生命周期）
        self.runtime_backend_resolver = builder.build_runtime_backend_resolver()
        self.agent_info = ctx.agent_config.agent_info if ctx.agent_config else None

        if chat.agent_cls is not None:
            self.agent_cls = chat.agent_cls
        if ctx.session_code is not None:
            self.thread_id = ctx.session_code
        if ctx.event_handler is not None:
            self.event_handler = ctx.event_handler
        return self

    def execute(self, execute_kwargs: ExecuteKwargs) -> Generator[str, None, None] | str:
        self.migration_v1()
        if not self.messages:
            self.messages = self.convert_history_to_messages()
        messages = self.messages
        self.chat_model.callbacks = self.callbacks
        return self._execute(messages, execute_kwargs)

    def stop(self):
        helper = GeneratorStreamingHelper(
            thread_id=self.thread_id,
        )
        if not helper.message_handler.is_cancel_requested(self.thread_id):
            logger.info(f"[STOP_DEBUG] Calling message_handler.request_cancel() for thread_id={self.thread_id}")
            helper.message_handler.request_cancel(self.thread_id)
        else:
            logger.info(f"[STOP_DEBUG] Cancel already requested for thread_id={self.thread_id}")
        # 用户主动停止时也释放资源
        self.release_resources()

    def release_resources(self) -> None:
        """释放 agent 持有的资源（沙箱后端等）。

        调用 RuntimeBackendResolver.close() 关闭所有已解析的沙箱后端。
        此方法是幂等的 — 多次调用不会产生副作用。
        """
        if self.runtime_backend_resolver is not None:
            try:
                self.runtime_backend_resolver.close()
            except Exception:
                logger.warning("ChatCompletionAgent.release_resources: 关闭沙箱资源失败", exc_info=True)
            finally:
                self.runtime_backend_resolver = None

    def convert_history_to_messages(self) -> list[BaseMessage]:
        if not self.chat_history:
            return []
        return self._chat_history_to_langchain_messages(self._convert_contents(self.chat_history))

    @property
    def model_name(self) -> str:
        return getattr(self.chat_model, "model_name", "")

    # ---------- 内部方法 ----------

    def _sync_checkpoint_messages(self, agent_e: Runnable, cfg: RunnableConfig) -> list[BaseMessage]:
        """同步 checkpoint 中的消息，返回 checkpoint 中非系统消息列表。

        使用 RemoveMessage 清除旧的 checkpoint 消息，避免与平台消息重复/冲突。
        这是保持 thread_id 稳定的前提条件（替代原来的 uuid4 后缀方案）。

        注意：此同步必须在 _execute 中执行（而非 prepare_stream），
        因为非流式路径（ainvoke）不经过 prepare_stream。
        """
        try:
            agent_state = run_coro_sync(agent_e.aget_state(cfg))
            checkpoint_messages = agent_state.values.get("messages", [])
            non_system_checkpoint_msgs = [m for m in checkpoint_messages if not isinstance(m, SystemMessage)]

            if non_system_checkpoint_msgs:
                remove_ops = []
                skipped_none_id_count = 0
                for m in non_system_checkpoint_msgs:
                    if m.id is not None:
                        remove_ops.append(RemoveMessage(id=m.id))
                    else:
                        skipped_none_id_count += 1

                if skipped_none_id_count > 0:
                    logger.warning(
                        "sync_checkpoint: %d checkpoint messages have id=None, "
                        "cannot remove via RemoveMessage, thread_id=%s",
                        skipped_none_id_count,
                        self.thread_id,
                    )

                if remove_ops:
                    run_coro_sync(
                        agent_e.aupdate_state(
                            cfg,
                            {"messages": remove_ops},
                            as_node="__start__",
                        )
                    )

            return non_system_checkpoint_msgs
        except Exception:
            logger.warning(
                "sync_checkpoint: failed to sync checkpoint messages, thread_id=%s",
                self.thread_id,
                exc_info=True,
            )
            return []

    def _update_aidev_agent_header(self, execute_kwargs: ExecuteKwargs) -> None:
        """Build complete X-BKAIDEV-Attributes header (agent.info + session) and inject in-place."""
        agent_info = self.agent_info or {}
        langgraph_thread_id = execute_kwargs.session_code or self.thread_id
        attrs = {
            "agent.info.code": agent_info.get("agent_code") or "",
            "agent.info.name": agent_info.get("agent_name") or "",
            "agent.info.service_catalogue": agent_info.get("service_catalogue") or "",
            "agent.info.sdk_version": pkg_version("aidev_agent"),
            "agent.session.caller_bk_app_code": execute_kwargs.caller_bk_app_code or "",
            "agent.session.caller_bk_biz_env": execute_kwargs.caller_bk_biz_env or "",
            "agent.session.caller_bk_biz_id": str(execute_kwargs.caller_bk_biz_id or ""),
            "agent.session.caller_executor": execute_kwargs.caller_executor or "",
            "agent.session.executor": execute_kwargs.executor or "",
            "agent.session.caller_order_type": execute_kwargs.caller_order_type or "",
            "agent.session.session_code": execute_kwargs.session_code or "",
            "agent.session.langgraph_thread_id": langgraph_thread_id,
        }

        header_value = json.dumps(attrs, ensure_ascii=True)
        for model in (self.chat_model, self.chat_model_non_thinking):
            if model is None or not hasattr(model, "default_headers"):
                continue
            if model.default_headers is None:
                model.default_headers = {}
            model.default_headers["X-BKAIDEV-Attributes"] = header_value

    def _fetch_platform_pv(self) -> list[dict]:
        """从平台拉取已存在的 sandbox PV，返回包含 PV 信息的列表。

        通过 resource_manager.retrieve_chat_session 获取会话的
        session_property.sandbox_pv_id，若存在则构造 platform 来源的 PV 条目；
        失败时返回空列表，不阻塞图执行。
        """
        if self.resource_manager is None or not self.thread_id:
            return []
        try:
            session = self.resource_manager.retrieve_chat_session(self.thread_id)
            session_property = (session or {}).get("session_property") or {}
            sandbox_pv_id = session_property.get("sandbox_pv_id")
        except Exception:
            logger.warning("restore platform PV failed: session_code=%s", self.thread_id, exc_info=True)
            return []
        if not sandbox_pv_id:
            return []
        return [
            {
                "type": "paas-sbx-pv",
                "volume_id": sandbox_pv_id,
                "volume_name": f"agent-pv-{self.thread_id}",
                "mount_path": "session",
                "source": "platform",
            }
        ]

    def _execute(self, messages: list[BaseMessage], execute_kwargs: ExecuteKwargs):
        if not messages:
            raise ValueError("The messages list cannot be empty.")
        self._update_aidev_agent_header(execute_kwargs)
        agent_e, cfg = self._get_agent(messages, execute_kwargs=execute_kwargs)
        cfg.setdefault("configurable", {})
        cfg["configurable"]["thread_id"] = self.thread_id
        cfg["configurable"]["execute_kwargs"] = execute_kwargs

        # 清除 checkpoint 中的旧消息，避免与平台消息重复
        # 平台传入的 messages 已包含完整历史，无需拼接
        self._sync_checkpoint_messages(agent_e, cfg)

        if execute_kwargs.stream:
            if execute_kwargs.legacy_streaming:
                return self._stream_with_legacy(agent_e, cfg, messages)
            else:
                return self._stream(agent_e, cfg, messages, execute_kwargs)

        else:
            try:
                platform_pv = self._fetch_platform_pv()
                input_state: dict[str, Any] = {"messages": messages, "execute_kwargs": execute_kwargs}
                if platform_pv:
                    input_state["runtime_paas_sbx_pv"] = platform_pv
                result = run_coro_sync(
                    agent_e.ainvoke(input_state, cfg),
                    timeout=execute_kwargs.invoke_timeout,
                )
                result_output = result.get("messages")[-1]
                return_data = {
                    "choices": [{"delta": {"role": "assistant", "content": result_output.content}}],
                    "model": self.model_name,
                    "id": result_output.id,
                    "reference_doc": result.get("reference_doc", []),
                }
                return return_data
            except Exception as e:
                logger.exception(f"Error executing agent: {e}")
                raise AgentException(message=f"Error executing agent: {e}")
            finally:
                # 非流式执行结束后释放资源
                self.release_resources()

    def _stream_with_legacy(
        self, agent_e: Runnable, cfg: RunnableConfig, messages: list[BaseMessage]
    ) -> Generator[Any, None, None]:
        platform_pv = self._fetch_platform_pv()
        _input: dict[str, Any] = {"messages": messages}
        if platform_pv:
            _input["runtime_paas_sbx_pv"] = platform_pv
        return agent_e.agent.stream_standard_event(agent_e, cfg, _input)

    def _stream(
        self, agent_e: Runnable, cfg: RunnableConfig, messages: list[BaseMessage], execute_kwargs: ExecuteKwargs
    ) -> Generator[Any, None, None]:
        # self.thread_id 是 ChatCompletionAgent 唯一的 thread_id 来源，
        # 同时用于 LangGraph checkpointer（跨请求持久化 state 如 PV）和 RabbitMQ 队列标识（断点续传）
        # 消息同步通过 RemoveMessage 在 _execute 中处理
        platform_pv = self._fetch_platform_pv()
        state = {}
        if platform_pv:
            state["runtime_paas_sbx_pv"] = platform_pv
        body = {
            "thread_id": self.thread_id,
            "run_id": messages[-1].id or uuid.uuid4().hex,
            "state": state,
            "messages": langchain_messages_to_agui(messages),
        }
        agent_input = AgentInput(**body)

        # 取消信号传递：cancel_checker 在用户点停止时由 Agent 内部轮询，
        # 触发后由 Agent 优雅发送 RunFinishedEvent
        def make_cancel_checker(thread_id: str):
            def cancel_checker() -> bool:
                return GeneratorStreamingHelper.is_cancelled(thread_id)

            return cancel_checker

        if isinstance(self.event_handler, BaseSessionWriter):
            if execute_kwargs.turn_id:
                self.event_handler.turn_id = execute_kwargs.turn_id
            self.event_handler.set_tools(self.tools)
        agui_entry = AidevAGUIAgent(
            name="test_agui_agent",
            graph=agent_e,
            event_handler=self.event_handler,
            config=cfg,
            tools={each.name: each for each in self.tools} if self.tools else {},
            cancel_checker=make_cancel_checker(self.thread_id),
            mcp_fetch_failures=getattr(self, "mcp_fetch_failures", []) or [],
        )

        return self._stream_with_queue(agui_entry, agent_input, queue_thread_id=self.thread_id)

    def _stream_with_queue(
        self, agui_entry: AidevAGUIAgent, agent_input: AgentInput, queue_thread_id: str | None = None
    ) -> Generator[Any, None, None]:
        """使用队列处理器缓存流式请求，支持断点续传

        断点续传逻辑：
        1. 如果流正在运行或已完成，且有未消费的缓存消息，则从 client_index 位置续传
        2. 否则开始新的流式请求

        客户端在消费时应该在每次成功接收后更新 client_index，
        以便在连接中断后能够从正确的位置继续。

        Args:
            agui_entry: AGUI Agent 入口
            agent_input: Agent 输入参数
            queue_thread_id: 队列标识ID（用于断点续传），默认使用 agent_input.thread_id

        Yields:
            流式响应数据
        """
        helper = GeneratorStreamingHelper(
            thread_id=queue_thread_id or agent_input.thread_id,
        )
        return helper.stream(async_to_sync_generator(agui_entry.run(agent_input)), on_complete=self._on_complete)

    def _on_complete(self):
        if self.event_handler and hasattr(self.event_handler, "set_streaming_finished"):
            self.event_handler.set_streaming_finished()
        # 流式执行结束后释放资源
        self.release_resources()

    def migration_v1(self) -> None:
        """兼容 v1 旧构造参数，统一迁移到当前运行时协议。"""
        if not isinstance(self.model_context_options, ModelContextSettings):
            self.model_context_options = migration_model_context_options_from_agent_options_v1(self.agent_options)
        if not isinstance(self.knowledge_query_options, KnowledgeSettings):
            self.knowledge_query_options = migration_knowledge_query_options_from_agent_options_v1(self.agent_options)
        if not isinstance(self.chat_model_non_thinking, BaseChatModel):
            self.chat_model_non_thinking = migration_chat_model_non_thinking_from_non_thinking_llm_v1(
                self.non_thinking_llm,
            )

    def _get_agent(
        self, messages: list[BaseMessage], *, execute_kwargs: ExecuteKwargs
    ) -> tuple[Runnable, RunnableConfig]:
        """
        由于在流式的时候，Response 立即返回会导致 trace 断掉，所以在_get_agent中添加 execute_kwargs
        execute_kwargs 有携带了 trace 上下文，以便于不要让 trace 断掉
        """
        # 合并 knowledge_bases / knowledge_items 到 knowledge_query_options
        if self.knowledge_bases:
            self.knowledge_query_options.knowledge_bases = self.knowledge_bases
        if self.knowledges:
            self.knowledge_query_options.knowledge_items = self.knowledges
        logger.info(f"callbacks: {self.callbacks}")
        return self.agent_cls.get_agent_executor(
            llm=self.chat_model,
            non_thinking_llm=self.chat_model_non_thinking or self.chat_model,
            extra_tools=self.tools,
            chat_history=messages[:-1],
            tool_execution_interval=self.TOOL_EXECUTION_INTERVAL,
            support_vision=self.support_vision,
            file_store=self.file_store,
            callbacks=self.callbacks,
            knowledge_query_options=self.knowledge_query_options,
            model_context_options=self.model_context_options,
            skills=self.skills,
            executor_info=self.executor_info,
            execute_kwargs=execute_kwargs,
            checkpointer=self.checkpointer if self.checkpointer else MemorySaver(),
            resource_manager=self.resource_manager,
            runtime_backend_resolver=self.runtime_backend_resolver,
        )

    def _chat_history_to_langchain_messages(self, chat_history: list[ChatPrompt]) -> list[BaseMessage]:
        """
        将 ChatPrompt 列表转换为 LangChain 消息列表
        支持从 builtin_property 中提取 tool_calls 和 tool_call_id 等协议字段，
        以支持多轮工具调用场景的历史消息透传。
        """
        messages: list[BaseMessage] = []
        for each in chat_history:
            bp = each.builtin_property or {}
            match each.role:
                case PromptRole.USER.value:
                    if isinstance(each.content, list):
                        new_content = []
                        for each_content in each.content:
                            if each_content.get("url"):
                                new_content.append({"type": "image_url", "image_url": {"url": each_content.get("url")}})
                            else:
                                new_content.append(each_content)
                        each.content = new_content
                        messages.append(HumanMessage(id=each.id, content=each.content))
                    else:
                        messages.append(HumanMessage(id=each.id, content=str(each.content)))
                case PromptRole.ASSISTANT.value | PromptRole.AI.value:
                    tool_calls = self._extract_tool_calls(bp)
                    messages.append(AIMessage(id=each.id, content=each.content, tool_calls=tool_calls))
                case PromptRole.SYSTEM.value:
                    messages.append(SystemMessage(id=each.id, content=each.content))
                case PromptRole.TOOL.value:
                    content = each.content if isinstance(each.content, str) else str(each.content)
                    messages.append(ToolMessage(id=each.id, content=content, tool_call_id=bp.get("tool_call_id", "")))
        return messages

    def _extract_tool_calls(self, builtin_property: dict) -> list[dict]:
        """从 builtin_property 中提取 tool_calls 列表

        注意：arguments 在数据库中存储为 JSON 字符串，需要解析为字典
        """
        tool_calls = []
        for tc in builtin_property.get("tool_calls", []):
            args_str = tc.get("function", {}).get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}

            tool_calls.append(
                {
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "args": args,
                    "type": "tool_call",
                }
            )
        return tool_calls

    def _convert_contents(self, contents: list[ChatPrompt]) -> list[ChatPrompt]:
        """将无需送到大模型处理的 content 去掉"""
        new_contents = []
        for each in contents:
            each.role = each.role.replace("hidden-", "")
            if each.role in self.SKIP_PROMPT_ROLE:
                continue
            if each.role == PromptRole.HIDDEN.value:
                each.role = PromptRole.USER.value
            if each.role == PromptRole.PAUSE.value:
                each.role = PromptRole.ASSISTANT.value
            if each.role == PromptRole.USER_IMAGE.value:
                if not self.support_vision:
                    raise AgentException(message="当前模型不支持图片识别,请切换其他模型")
                each.role = PromptRole.USER.value
                match = self.IMAGE_FILE_PATTERN.search(each.content)
                if match:
                    file_path, _ = match.group(1), match.group(2)
                    each.content = [{"type": "image_url", "image_url": {"url": file_path}}]
                    # 图片不计算实际大小，但不能为 0 —— 给一个大于 0 的占位值
                    self.files.append({"file_name": file_path, "file_size": 100})
                else:
                    raise AgentException(message="图片md格式非法")
            # deepseek-r1 系列不支持 system role，需要降级为 user
            if each.role == PromptRole.SYSTEM.value and "deepseek-r1" in self.model_name:
                each.role = PromptRole.USER.value
            new_contents.append(each)

        return new_contents


class ChatAgentBuilder:
    """``ChatCompletionAgent`` 装配器

    把原 ``AgentInstanceFactory`` 中所有 Chat 专属装配方法收敛在这里：

    - 模型 / 工具 / 知识 / 技能装配
    - 聊天历史构建（含 tool_calls 过滤、think 移除、role_history 拼接、modify_last_system_message）
    - executor_info / checkpointer 取值
    - model_context_options / knowledge_query_options 取值
    - handle_agent_switch（替换 system 消息）
    - specific_resources 提取（``_handle_last_human_message``）

    持有 :class:`AgentBuildContext` 引用：
    - 主智能体配置直接读 ``self.ctx.agent_config``（已在 ctx 装配阶段一次性预读）。
    - 通用字段读 ``self.ctx.{resource_manager, username, agent_code, session_context_data, switch_agent}``。
    - Chat 专属字段读 ``self.ctx.chat.{temperature, max_tokens, auth_headers, checkpointer, ...}``。
    """

    def __init__(self, ctx: AgentBuildContext):
        self.ctx = ctx
        self._specific_resources: list[dict] = []
        self._mcp_fetch_failures: list[dict] = []
        self._executor_info: dict | None = None
        self._runtime_backend_resolver: Any | None = None
        # 装配前先从最后一条 user 消息提取 specific_resources，供 build_tools / build_knowledge_bases 过滤
        self._handle_last_human_message(ctx.session_context_data)

    @property
    def mcp_fetch_failures(self) -> list[dict]:
        """MCP 工具拉取失败记录，由 ``build_tools`` 写入"""
        return self._mcp_fetch_failures

    def build_runtime_backend_resolver(self) -> RuntimeBackendResolver:
        """构造 RuntimeBackendResolver。

        resolver 将通过 AgentExecutorKwargs 传入 ReActAgentBuilder。
        """
        self._runtime_backend_resolver = RuntimeBackendResolver(default_runtime="local")
        return self._runtime_backend_resolver

    # ---------- 公共：装配方法（被 ChatCompletionAgent.build 调用） ----------

    def build_chat_model(self) -> BaseChatModel:
        """构建聊天模型"""
        config = self.ctx.agent_config
        chat = self.ctx.chat or ChatBuildExtras()

        if not config.chat_model:
            raise ValueError("请配置智能体默认模型并重新发布")

        kwargs: dict[str, Any] = {
            "model": config.chat_model,
            "base_url": settings.LLM_GW_ENDPOINT,
        }

        temperature = chat.temperature if chat.temperature is not None else config.temperature
        if temperature is not None:
            kwargs["temperature"] = temperature

        max_tokens = chat.max_tokens if chat.max_tokens is not None else config.max_tokens
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if chat.auth_headers:
            kwargs["auth_headers"] = chat.auth_headers

        if chat.default_headers:
            kwargs["default_headers"] = chat.default_headers

        if self.ctx.session_code:
            kwargs["session_code"] = self.ctx.session_code

        return ChatModel.get_setup_instance(**kwargs)

    def build_chat_model_non_thinking(self) -> BaseChatModel | None:
        """构建非思考模型 (返回 ChatModel 实例)"""
        model_name = self.ctx.agent_config.non_thinking_llm
        if not model_name:
            return None
        kwargs: dict[str, Any] = {
            "model": model_name,
            "base_url": settings.LLM_GW_ENDPOINT,
        }
        chat = self.ctx.chat or ChatBuildExtras()
        if chat.auth_headers:
            kwargs["auth_headers"] = chat.auth_headers
        return ChatModel.get_setup_instance(**kwargs)

    def build_chat_history(self, session_context_data: List[dict]) -> List[ChatPrompt]:
        """构建聊天历史"""
        config = self.ctx.agent_config
        role_prompt_roles = {
            PromptRole.USER.value,
            PromptRole.ASSISTANT.value,
            PromptRole.SYSTEM.value,
            PromptRole.PAUSE.value,
            "hidden-user",
            "hidden-assistant",
            "hidden-system",
        }
        role_history = [
            ChatPrompt(role=each["role"].replace("hidden-", ""), content=each["content"])
            for each in (config.role_prompts or [])
            if each.get("content") and each.get("role") in role_prompt_roles
        ]

        chat_history = [
            ChatPrompt.model_validate(each)
            for each in (session_context_data or [])
            if each.get("content") and each["role"] != "system"
        ]
        for each in chat_history:
            if each.role != "assistant":
                continue
            each.content = self._remove_think(each.content)

        chat_history = self._filter_unmatched_tool_calls(chat_history)

        self._modify_last_system_message(chat_history)
        chat_history = role_history + chat_history
        return chat_history

    def build_non_thinking_llm(self) -> str | None:
        """构建非思考模型

        .. deprecated::
            Use :meth:`build_chat_model_non_thinking` instead.
        """
        import warnings

        warnings.warn(
            "build_non_thinking_llm is deprecated, use build_chat_model_non_thinking instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.ctx.agent_config.non_thinking_llm

    def build_knowledge_bases(self) -> List[dict]:
        """构建知识库"""
        config = self.ctx.agent_config
        specific_resources = [
            each.get("id") for each in self._specific_resources if each.get("type") == "knowledgebase"
        ]
        if specific_resources:
            knowledgebase_ids = [
                each for each in config.knowledgebase_ids if specific_resources and each in specific_resources
            ]
        else:
            knowledgebase_ids = config.knowledgebase_ids
        logger.info(
            f"ChatAgentBuilder: config knowledgebase_ids->[{config.knowledgebase_ids}], "
            f"specific_resources->[{specific_resources}]"
        )
        return [self.ctx.resource_manager.retrieve_knowledgebase(_id) for _id in knowledgebase_ids]

    def build_knowledge_items(self) -> List[dict]:
        """构建知识条目"""
        config = self.ctx.agent_config
        return [self.ctx.resource_manager.retrieve_knowledge(_id) for _id in config.knowledge_ids]

    def build_tools(self) -> List[Any]:
        """构建工具"""
        config = self.ctx.agent_config
        specific_mcps = [each.get("code") for each in self._specific_resources if each.get("type") == "mcp"]
        if specific_mcps:
            mcp_server_config = {each: config.mcp_server_config.get(each) for each in specific_mcps}
        else:
            mcp_server_config = config.mcp_server_config
        mcp_result = self.ctx.resource_manager.construct_mcp(
            mcp_config=mcp_server_config,
            username=self.ctx.username,
            executor_info=self._executor_info,
        )
        self._mcp_fetch_failures = [f.model_dump() for f in mcp_result.fetch_failures]
        logger.info(f"ChatAgentBuilder: mcp_server_config->[{mcp_server_config}]")
        specific_tools = [each.get("code") for each in self._specific_resources if each.get("type") == "tool"]
        if specific_tools:
            tool_codes = [each for each in config.tool_codes if each in specific_tools]
        else:
            tool_codes = config.tool_codes
        logger.info(f"ChatAgentBuilder: tool_codes->[{tool_codes}]")
        return [self.ctx.resource_manager.construct_tool(tool_code) for tool_code in tool_codes] + mcp_result.tools

    def build_skills(self) -> list | None:
        """构建关联技能"""
        return self.ctx.agent_config.related_skills

    def build_knowledge_query_options(self):
        """从 AgentConfig 构建 KnowledgeSettings；新协议为空时兼容旧 agent_options。"""
        data = self.ctx.agent_config.knowledge_query_options_data
        if data:
            return KnowledgeSettings.model_validate(data)
        return migration_knowledge_query_options_from_agent_options_v1(self.ctx.agent_config.agent_options)

    def build_model_context_options(self) -> ModelContextSettings | None:
        """从 AgentConfig 构建 ModelContextSettings；新协议为空时兼容旧 agent_options。"""
        data = self.ctx.agent_config.model_context_options_data
        if data:
            return ModelContextSettings.model_validate(data)
        return migration_model_context_options_from_agent_options_v1(self.ctx.agent_config.agent_options)

    def build_support_vision(self) -> bool:
        """从 prompt_setting.support_upload.vision 构建 support_vision"""
        support_upload = self.ctx.agent_config.model_context_options_data.get("support_upload") or {}
        return bool(support_upload.get("vision", False))

    def build_executor_info(self) -> dict:
        """构建执行用户信息，包含 access_token / app_code / app_secret 用于沙箱认证和 MCP 调用"""
        info = {"executor": self.ctx.username}
        access_token = self.ctx.resource_manager.resolve_access_token(self.ctx.username)
        if access_token:
            info["access_token"] = access_token
        # 将 resource_manager 的 app_code/app_secret 传入 executor_info，
        # 用于 PaaS Sandbox API 的应用态认证（平台测试页进程是平台凭证）
        if self.ctx.resource_manager.app_code:
            info["app_code"] = self.ctx.resource_manager.app_code
        if self.ctx.resource_manager.app_secret:
            info["app_secret"] = self.ctx.resource_manager.app_secret
        logger.info(
            f"[credential] build_executor_info: username={self.ctx.username}, "
            f"access_token={'***' if access_token else 'empty'}, "
            f"app_code={info.get('app_code', 'empty')}, "
            f"has_app_secret={bool(info.get('app_secret'))}, "
            f"rm_type={type(self.ctx.resource_manager).__name__}"
        )
        self._executor_info = info
        return info

    def build_checkpointer(self) -> BaseCheckpointSaver:
        """获取 Checkpointer，必须注入，否则抛出异常"""
        chat = self.ctx.chat or ChatBuildExtras()
        if chat.checkpointer is not None:
            return chat.checkpointer
        raise ValueError("Checkpointer is required but not provided. Please inject a valid checkpointer.")

    def get_role_prompt(self) -> str | None:
        """获取角色提示词"""
        config = self.ctx.agent_config
        return config.role_prompts[0]["content"] if config.role_prompts else None

    def handle_agent_switch(self) -> None:
        """处理智能体切换：替换最后一条 system 消息为新 role_prompt"""
        if not self.ctx.switch_agent:
            return

        logger.info(f"ChatAgentBuilder: switching agent to->[{self.ctx.agent_code}]")
        for item in reversed(self.ctx.session_context_data):
            if item["role"] == "system":
                item["content"] = self.get_role_prompt()
                break

    # ---------- 内部方法 ----------

    def _handle_last_human_message(self, session_context_data: List[dict]):
        """处理最后一条 human 消息，判断有 resources"""
        if not session_context_data:
            return

        for item in reversed(session_context_data):
            logger.info(
                f"ChatAgentBuilder: handling last human message with resources in session_context_data->[{item}]"
            )
            if item.get("role") == PromptRole.USER.value:
                extra = item.get("extra") or {}
                if extra.get("resources"):
                    self._specific_resources = extra.get("resources")
                break

    def _filter_unmatched_tool_calls(self, chat_history: List[ChatPrompt]) -> List[ChatPrompt]:
        """过滤没有匹配工具结果的 assistant 消息

        当 assistant 消息包含 tool_calls 但没有对应的 tool 结果消息时，
        该 assistant 消息会导致模型调用失败（模型期望每个 tool_use 都有对应的 tool_result）。

        Args:
            chat_history: 聊天历史列表

        Returns:
            过滤后的聊天历史列表，移除了不完整的工具调用链
        """
        if not chat_history:
            return chat_history

        tool_result_ids: set[str] = set()
        for prompt in chat_history:
            if prompt.role == "tool":
                tool_call_id = prompt.builtin_property.get("tool_call_id", "")
                if tool_call_id:
                    tool_result_ids.add(tool_call_id)

        # 过滤 assistant 消息中未匹配的 tool_calls：
        # 全部无结果 → 整条丢弃；部分有结果 → 仅保留匹配的 tool_calls。
        filtered_history: List[ChatPrompt] = []
        for prompt in chat_history:
            if prompt.role != "assistant":
                filtered_history.append(prompt)
                continue

            tool_calls = self._extract_tool_calls_from_prompt(prompt)

            if not tool_calls:
                filtered_history.append(prompt)
                continue

            matched_calls = [tc for tc in tool_calls if tc.get("id", "") in tool_result_ids]
            unmatched_calls = [tc for tc in tool_calls if tc.get("id", "") not in tool_result_ids]

            if not matched_calls:
                logger.info(
                    f"ChatAgentBuilder: filtering assistant message with no matched tool_calls, "
                    f"message_id=[{prompt.id}], tool_calls_count=[{len(tool_calls)}]"
                )
                continue

            if unmatched_calls:
                logger.info(
                    f"ChatAgentBuilder: removing unmatched tool_calls from assistant message, "
                    f"message_id=[{prompt.id}], total_calls=[{len(tool_calls)}], "
                    f"matched=[{len(matched_calls)}], unmatched=[{len(unmatched_calls)}]"
                )
                self._update_tool_calls_in_prompt(prompt, matched_calls)

            filtered_history.append(prompt)

        return filtered_history

    def _extract_tool_calls_from_prompt(self, prompt: ChatPrompt) -> List[dict]:
        """从 ChatPrompt 中提取 tool_calls 列表

        Args:
            prompt: ChatPrompt 对象

        Returns:
            tool_calls 列表，每个元素包含 id, name, args 字段
        """
        builtin_property = prompt.builtin_property or {}
        tool_calls_raw = builtin_property.get("tool_calls", [])

        if not tool_calls_raw:
            return []

        tool_calls = []
        for tc in tool_calls_raw:
            # arguments 在数据库中存储为 JSON 字符串，需要解析为字典
            args_str = tc.get("function", {}).get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}

            tool_calls.append(
                {
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "args": args,
                    "type": "tool_call",
                }
            )

        return tool_calls

    def _update_tool_calls_in_prompt(self, prompt: ChatPrompt, matched_tool_calls: List[dict]) -> None:
        """更新 ChatPrompt 中的 tool_calls，只保留匹配的调用

        Args:
            prompt: ChatPrompt 对象
            matched_tool_calls: 匹配的 tool_calls 列表（来自 _extract_tool_calls_from_prompt 的格式）
        """
        builtin_property = prompt.builtin_property or {}
        tool_calls_raw = builtin_property.get("tool_calls", [])

        matched_ids = {tc.get("id", "") for tc in matched_tool_calls}

        filtered_tool_calls = [tc for tc in tool_calls_raw if tc.get("id", "") in matched_ids]

        if builtin_property:
            builtin_property["tool_calls"] = filtered_tool_calls
        else:
            prompt.builtin_property = {"tool_calls": filtered_tool_calls}

    def _modify_last_system_message(self, chat_history: List[ChatPrompt]) -> None:
        if not self.ctx.agent_code:
            return

        role_prompt = self.get_role_prompt()
        if not role_prompt:
            return

        for prompt in reversed(chat_history):
            if prompt.role == "system":
                prompt.content = role_prompt
                break

    @staticmethod
    def _remove_think(content: str) -> str:
        """移除 HTML 中的思考部分内容

        Args:
            content: 包含思考内容的 HTML 字符串

        Returns:
            清理后的内容字符串
        """
        _content = re.sub(
            r'<section class="think-head click-close">[\s\S]*?</section>',
            "",
            content,
            flags=re.DOTALL,
        )

        _content = re.sub(
            r'<section class="think-head click-close closed">[\s\S]*?</section>',
            "",
            _content,
            flags=re.DOTALL,
        )

        _content = re.sub(r'<section class="think-body">[\s\S]*?</section>', "", _content, flags=re.DOTALL)

        if not _content.strip():
            think_body_match = re.search(r'<section class="think-body">([\s\S]*?)</section>', content, re.DOTALL)
            if think_body_match:
                _content = think_body_match.group(1).strip()

        return _content.strip()
