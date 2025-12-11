# -*- coding: utf-8 -*-
"""
TencentBlueKing is pleased to support the open source community by making
蓝鲸智云 - AIDev (BlueKing - AIDev) available.
Copyright (C) 2025 THL A29 Limited,
a Tencent company. All rights reserved.
Licensed under the MIT License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the License for the
specific language governing permissions and limitations under the License.
We undertake not to change the open source license (MIT license) applicable
to the current version of the project delivered to anyone in the future.
"""
from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional, Tuple
from typing import TYPE_CHECKING, Annotated

from langchain.agents.middleware.types import (
    AgentState,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.stores import ByteStore
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import add_messages
from langgraph.graph.state import StateGraph
from langgraph.store.memory import InMemoryStore
from typing_extensions import Literal, TypedDict, TypeVar

from aidev_agent.core.nodes.context_processor import ContextProcessor
from aidev_agent.core.nodes.model import build_model_node as std_make_model_node
from aidev_agent.core.nodes.tool import build_tool_node
from aidev_agent.enums import Decision
from aidev_agent.packages.langchain_core.models.utils import is_model_without_function_calling
from aidev_agent.packages.langchain_core.tools.builtin import add_image_to_chat_context
from aidev_agent.packages.langgraph.streaming.streaming_protocol import AgentStreamAdapter
from aidev_agent.services.pydantic_models import AgentOptions

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable
    from langgraph.store.base import BaseStore
ResponseT = TypeVar("ResponseT")

logger = logging.getLogger(__name__)


class DefaultState(TypedDict):
    # 消息历史（包含人类消息、AI消息和工具执行结果）
    messages: Annotated[List[BaseMessage], add_messages]
    # 知识库相关数据
    decision: Decision
    knowledge_resources_highly_relevant: list
    knowledge_resources_moderately_relevant: list
    knowledge_resources_lowly_relevant: list
    reference_doc: list
    knowledge_content: list
    knowledge_qa_content: list
    with_qa_response: list


class KnowledgeInputState(TypedDict):
    input: str
    query: str


class ReActAgent:
    """LangGraph V1 QA Agent 构建器。

    支持工具调用的完整 ReAct 循环：
    - model 节点：调用 LLM 进行推理
    - tool 节点：执行工具调用
    - 条件路由：model → tool / END，tool → model
    """
    @staticmethod
    def _prepare_agent_options(
        agent_options: Optional[AgentOptions],
        *,
        knowledge_items: Optional[List[Dict]] = None,
        knowledge_bases: Optional[List[Dict]] = None,
        role_prompt: Optional[str] = None,
        intent_recognition_kwargs: Optional[Dict] = None,
    ) -> AgentOptions:
        options = agent_options or AgentOptions()

        ir_options = options.intent_recognition_options
        kq_options = options.knowledge_query_options
        if intent_recognition_kwargs:
            if "tool_output_compress_thrd" in intent_recognition_kwargs:
                # aidev_agent/services/pydantic_models.py 中，默认配置为 5000
                ir_options.tool_output_compress_thrd = intent_recognition_kwargs["tool_output_compress_thrd"]
            if "token_limit_margin" in intent_recognition_kwargs:
                # aidev_agent/services/pydantic_models.py 中，默认配置为 100
                kq_options.token_limit_margin = intent_recognition_kwargs["token_limit_margin"]
            if "max_tool_output_len" in intent_recognition_kwargs:
                # aidev_agent/services/pydantic_models.py 中，默认配置为 500
                ir_options.max_tool_output_len = intent_recognition_kwargs["max_tool_output_len"]
        if knowledge_bases:
            kq_options.knowledge_bases = knowledge_bases
        if knowledge_items:
            kq_options.knowledge_items = knowledge_items
        if role_prompt:
            kq_options.role_prompt = role_prompt

        return options

    @staticmethod
    def _prepare_agent_tools(
        *,
        extra_tools: List[BaseTool] = None,
        support_vision: bool = False,
        ignore_errors: bool = False,
    ) -> List[BaseTool]:
        tools: List[BaseTool] = []
        if extra_tools:
            tools.extend(extra_tools or [])
        if support_vision:
            tools.append(add_image_to_chat_context)
        if ignore_errors:
            # NOTE: 在 StructuredChatAgent 中修改 tools 中的参数
            # 使得如果 LLM 调用工具时如果出现以下类型的错误，可以重新尝试，继续进行而不阻碍过程
            for i in range(len(tools)):
                tools[i].handle_validation_error = True
                tools[i].handle_tool_error = True
        return tools

    @staticmethod
    def _prepare_checkpointer(
        *,
        checkpointer: BaseCheckpointSaver | None = None
    ):
        if isinstance(checkpointer, BaseCheckpointSaver):
            return checkpointer
        return MemorySaver()

    @staticmethod
    def _prepare_store(
        *,
        store: "BaseStore | None",
        file_store: Optional[ByteStore],
    ) -> "BaseStore":
        """使用 LangGraph Store 模拟 request_local.current_user_store。

        - 默认使用 InMemoryStore
        - 预先写入 file_store / image / knowledge_bases / knowledge_items / reference_doc
        """
        if store is None:
            store = InMemoryStore()
        return store

    @staticmethod
    def _should_continue(state: dict) -> Literal["tools", "end"]:
        """条件路由函数：决定 model 节点后的下一步。

        检查模型输出是否包含 tool_calls：
        - 如果有 tool_calls，路由到 tools 节点执行工具
        - 否则路由到 end 结束对话

        Args:
            state: 当前状态字典

        Returns:
            "tools" 或 "end"
        """
        messages = state.get("messages", [])
        if not messages:
            return "end"

        last_message = messages[-1]

        # 检查最后一条消息是否是 AIMessage 并且包含 tool_calls
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"

        return "end"

    @classmethod
    def _build_graph(
        cls,
        *,
        llm: BaseChatModel,
        knowledge_llm: BaseChatModel,
        tools: List[BaseTool],
        use_structured_response: bool,
        agent_options: AgentOptions,
        enable_query_clarification: Optional[bool],
        state_schema,
        callbacks: List,
        debug: bool,
        checkpointer,
        store,
        interrupt_before,
        interrupt_after,
        name,
        cache,
    ) -> Tuple["Runnable", RunnableConfig]:
        """构建 LangGraph 图。

        图结构：
        - 无知识库配置: START → model → tools/END
        - 有知识库配置: START → knowledge → model → tools/END
        - 如果有工具: model → (条件) → tools / END, tools → model

        Args:
            llm: 语言模型
            knowledge_llm: 知识库语言模型
            tools: 工具列表
            use_structured_response: 是否使用结构化输出模式
            agent_options: Agent 配置选项
            enable_query_clarification: 是否启用查询澄清
            state_schema: 状态模式
            callbacks: 回调列表
            debug: 是否开启调试模式
            checkpointer: 检查点
            store: 存储
            interrupt_before: 中断前节点列表
            interrupt_after: 中断后节点列表
            name: 图名称
            cache: 缓存

        Returns:
            (CompiledGraph, RunnableConfig) 元组
        """
        graph = StateGraph(state_schema=state_schema)

        # 处理 enable_query_clarification 的默认值
        if enable_query_clarification is None:
            model_name = getattr(llm, "model_name", "")
            enable_query_clarification = (
                model_name == "gpt-4o" or "deepseek" in model_name or "qwq" in model_name
            )

        # 从 agent_options 获取配置
        knowledge_query_options = agent_options.knowledge_query_options
        rejection_message = knowledge_query_options.rejection_message
        role_prompt = knowledge_query_options.role_prompt
        use_general_knowledge_on_miss = knowledge_query_options.is_response_when_no_knowledgebase_match

        # 检查是否配置了知识库
        has_knowledge = (
            knowledge_query_options.knowledge_bases
            or knowledge_query_options.knowledge_items
        )

        # 构建上下文处理器（传入 tools 参数）
        context_processor = ContextProcessor(
            use_structured_response=use_structured_response,
            enable_query_clarification=enable_query_clarification,
            rejection_message=rejection_message,
            role_prompt=role_prompt,
            use_general_knowledge_on_miss=use_general_knowledge_on_miss,
            tools=tools,
        )

        # 如果配置了知识库,添加 knowledge 节点
        if has_knowledge:
            from aidev_agent.core.nodes.knowledge import make_knowledge_node

            knowledge_node = make_knowledge_node(
                llm=knowledge_llm,
                agent_options=agent_options,
            )
            graph.add_node("knowledge", knowledge_node)
            graph.add_edge(START, "knowledge")

        # 创建模型节点（不再传入 tools 参数，改为通过 context_processor 获取）
        model_node = std_make_model_node(
            llm=llm,
            context_processor=context_processor,
            use_structured_response=use_structured_response,
        )

        # 添加模型节点
        graph.add_node("model", model_node)

        # 根据是否有知识库节点,连接不同的边
        if has_knowledge:
            # 有知识库: knowledge → model
            graph.add_edge("knowledge", "model")
        else:
            # 无知识库: START → model
            graph.add_edge(START, "model")

        # 如果有工具，添加工具节点和条件路由
        if tools:
            tool_node = build_tool_node(tools=tools)
            graph.add_node("tools", tool_node)
            # model → (should_continue) → tools / end
            graph.add_conditional_edges(
                "model",
                cls._should_continue,
                {
                    "tools": "tools",
                    "end": END,
                },
            )
            # tools → model (形成 ReAct 循环)
            graph.add_edge("tools", "model")
        else:
            # 无工具时直接结束
            graph.add_edge("model", END)

        compile_graph = graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug,
            name=name,
            cache=cache,
        )

        cfg = RunnableConfig()
        cfg["callbacks"] = callbacks
        cfg["configurable"] = {
            "thread_id": uuid.uuid4(),
            "agent_options": agent_options,
            "debug": debug,
        }

        return compile_graph, cfg

    @classmethod
    def get_agent_executor(
        cls,
        *,
        llm: BaseChatModel,
        knowledge_llm: BaseChatModel,
        non_thinking_llm: BaseChatModel | str = None,
        extra_tools: Optional[List[BaseTool]] = None,
        prefix: Optional[str] = None,
        role_prompt: Optional[str] = None,
        suffix: Optional[str] = None,
        format_instructions: Optional[str] = None,
        chat_history: Optional[List[BaseMessage]] = None,
        callbacks: Optional[List] = None,
        knowledge_items: Optional[List[Dict]] = None,
        knowledge_bases: Optional[List[Dict]] = None,
        file_store: Optional[ByteStore] = None,
        support_vision: bool = False,
        llm_token_limit=28000,
        agent_options: Optional[AgentOptions] = None,
        state_schema: type[AgentState[ResponseT]] | None = None,
        checkpointer: "Checkpointer | None" = None,
        store: "BaseStore | None" = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        debug: bool = False,
        name: str | None = None,
        cache: "BaseCache | None" = None,
        intent_recognition_kwargs=None,
        enable_query_clarification: Optional[bool] = None,
        **kwargs,
    ) -> Tuple["Runnable", RunnableConfig]:
        """创建 Agent 执行器。

        Args:
            llm: 语言模型
            knowledge_llm: 知识库语言模型
            extra_tools: 额外工具列表
            prefix: 系统提示前缀
            role_prompt: 角色提示
            suffix: 后缀（未使用）
            format_instructions: 格式指令（未使用）
            chat_history: 聊天历史
            callbacks: 回调列表
            knowledge_items: 知识项
            knowledge_bases: 知识库
            file_store: 文件存储
            support_vision: 是否支持视觉
            llm_token_limit: LLM token 限制
            agent_options: Agent 配置选项
            state_schema: 状态模式
            checkpointer: 检查点
            store: 存储
            interrupt_before: 中断前节点列表
            interrupt_after: 中断后节点列表
            debug: 是否开启调试模式
            name: 图名称
            cache: 缓存
            intent_recognition_kwargs: 意图识别参数
            enable_query_clarification: 是否启用查询澄清

        Returns:
            (CompiledGraph, RunnableConfig) 元组
        """
        callbacks = callbacks or []
        use_structured_response = is_model_without_function_calling(llm) and extra_tools
        # 统一处理 agent_options
        prepared_agent_options = cls._prepare_agent_options(
            agent_options,
            knowledge_items=knowledge_items,
            knowledge_bases=knowledge_bases,
            role_prompt=role_prompt,
            intent_recognition_kwargs=intent_recognition_kwargs,
        )
        # 统一处理 tools
        tool_ignore_errors = use_structured_response
        tools: List[BaseTool] = cls._prepare_agent_tools(
            extra_tools=extra_tools, support_vision=support_vision, ignore_errors=tool_ignore_errors
        )
        # 统一处理 checkpoint
        # checkpointer = cls._prepare_checkpointer(checkpointer=checkpointer)
        # 初始化 Store
        store = cls._prepare_store(store=store, file_store=file_store)
        # 定制 ReAct chat prompt template
        if state_schema is None:
            state_schema = DefaultState
        # 构建图
        compile_graph, cfg = cls._build_graph(
            llm=llm,
            knowledge_llm=knowledge_llm,
            tools=tools,
            use_structured_response=use_structured_response,
            agent_options=prepared_agent_options,
            enable_query_clarification=enable_query_clarification,
            state_schema=state_schema,
            callbacks=callbacks,
            debug=debug,
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            name=name,
            cache=cache,
        )
        # 添加适配器
        compile_graph.agent = AgentStreamAdapter()
        return compile_graph, cfg
