# -*- coding: utf-8 -*-
from __future__ import annotations

import dataclasses
import json
from logging import getLogger
from typing import Any

from aidev_agent.config import settings
from aidev_agent.core.tools.a2a_tools.progress import build_enriched_result, count_tool_calls, detect_intermediate_step
from aidev_agent.core.tools.a2a_tools.provider import _A2A_SUBAGENT_FLAG
from aidev_agent.core.tools.a2a_tools.types import AgentResult, AgentSpec, ExitReason
from aidev_agent.enums import SessionsStatus
from aidev_agent.pydantic_models import ExecuteKwargs

logger = getLogger(__name__)

_ERR_MISSING_AGENT_INSTANCE = "Missing agent_cls in spec.params"
_ERR_MISSING_CTX = "Missing ctx in spec.params"


class LocalBackend:
    """Local 本地后端：基于 ``agent_cls + ctx`` 构建并同步执行子 Agent。

    需要的 ``spec.params``：

    - ``agent_cls``：Agent 类（如 ``ChatCompletionAgent``），由
      ``ChatAgentBuilder.build_subagents`` 在构造 ``AgentSpec`` 时填充。
      LocalBackend 在运行时通过 ``agent_cls()`` 创建实例再 ``build(ctx)``。
    - ``ctx``：``AgentBuildContext``，用于装配子 Agent。

    运行时 kwargs：

    - ``config``：父 Agent 的 ``RunnableConfig``，从中获取 ``execute_kwargs`` 和 ``executor``。
    - ``session_code``：会话标识；必须非空（由调用方 Task/Member 保证）。
    """

    def prepare_session(self, rm: Any, agent_name: str, session_code: str, message: str, executor: str) -> list[dict]:
        """准备平台 session：创建、保存用户输入、加载历史。

        执行流程：
        1. 通过 resource_manager 创建/获取平台 Session（幂等）
        2. 设置 session 状态为 RUNNING
        3. 保存用户输入到 session_content
        4. 从平台加载历史会话上下文（包含刚写入的用户消息），过滤 system 角色

        返回的平台原始上下文列表（``session_context_data``）将通过
        ``ctx.session_context_data`` 传入 ``ChatCompletionAgent.build(ctx)``，
        复用 ``ChatAgentBuilder.build_chat_history`` → ``convert_history_to_messages``
        管道完成转换，无需在本层重复实现。

        Args:
            rm: resource_manager 实例
            agent_name: Agent 名称
            session_code: 会话标识
            message: 用户输入消息
            executor: 执行人用户名

        Returns:
            session_context_data 列表
        """
        session_context_data: list[dict] = []

        headers = {"X-BKAIDEV-USER": executor}
        session_type = "dev" if settings.BKPAAS_ENVIRONMENT.lower() in {"dev", "development"} else "agent"
        rm.get_or_create_session(
            session_code=session_code,
            session_name=f"local-{agent_name}-session",
            protocol_version="v2",
            is_temporary=settings.BKAI_A2A_SESSION_TEMPORARY,
            session_type=session_type,
            headers=headers,
        )
        rm.update_session_status(session_code, SessionsStatus.RUNNING.value)

        rm.save_session_content(
            session_code=session_code,
            role="user",
            content=message,
        )

        context = rm.get_chat_session_context(session_code) or []
        session_context_data = [each for each in context if each.get("role", "") != "system"]

        return session_context_data

    def execute(
        self, spec: AgentSpec, message: str, *, session_code: str = "", config: Any = None, **kwargs: Any
    ) -> AgentResult:
        """执行子 Agent 调用（统一入口 + 可观测性）。

        执行流程：
        1. 校验 session_code / agent_cls / ctx / rm 必须存在
        2. 从 config 获取 execute_kwargs 和 executor，构造子 Agent execute_kwargs
        3. 准备平台 session（创建、保存用户输入、加载历史）
        4. 更新 event_handler 的 session_code
        5. 注入 session_code + session_context_data 到 ctx + 注入嵌套保护标志（D-09）
        6. 实例化 agent_cls 并 build 子 Agent（历史消息由 build 内部管道处理）
        7. 执行子 Agent
        8. 更新 session 状态为 FINISHED

        Args:
            spec: Agent 规格，``params`` 必须含 ``agent_cls`` / ``ctx``
            message: 发给子 Agent 的消息文本
            session_code: 会话标识；必须非空（由调用方 Task/Member 保证）
            config: RunnableConfig，从中获取 execute_kwargs 和 executor
            **kwargs: 运行时参数（``progress_callback`` 等）

        Returns:
            AgentResult 标准化富结果（不可变 Pydantic 模型）
        """
        # 1. session_code 必须非空
        if not session_code:
            raise ValueError("session_code must not be empty for LocalBackend")

        # 2. 从 spec.params 获取 agent_cls、ctx
        agent_cls = spec.params.get("agent_cls")
        if agent_cls is None:
            raise ValueError(_ERR_MISSING_AGENT_INSTANCE)
        ctx = spec.params.get("ctx")
        if ctx is None:
            raise ValueError(_ERR_MISSING_CTX)

        # 3. 获取 rm
        rm = getattr(ctx, "resource_manager", None)
        if not rm:
            raise ValueError("ctx.resource_manager must not be empty for LocalBackend")

        # 4. 从 config 获取 execute_kwargs 和 executor，构造子 Agent execute_kwargs
        executor = ""
        caller_bk_app_code = kwargs.get("caller_bk_app_code", "")
        subagent_execute_kwargs = self._extract_execute_kwargs(
            config, session_code=session_code, caller_bk_app_code=caller_bk_app_code
        )
        if config and isinstance(config, dict):
            ek = config.get("configurable", {}).get("execute_kwargs")
            if ek and hasattr(ek, "executor"):
                executor = ek.executor or ""

        try:
            # 5. 准备平台 session（创建、保存用户输入、加载历史）
            session_context_data = self.prepare_session(rm, spec.name, session_code, message, executor)

            # 6. 更新 event_handler session_code（AGUISessionWriter 由 ChatAgentBuilder 构造）
            event_handler = getattr(ctx, "event_handler", None)
            if event_handler:
                event_handler.session_code = session_code

            # 7. 注入 session_code + session_context_data 到 ctx
            ctx = dataclasses.replace(ctx, session_code=session_code, session_context_data=session_context_data)

            # 8. 注入嵌套保护标志（D-09）
            existing_extra = dict(getattr(ctx, "extra", None) or {})
            existing_extra[_A2A_SUBAGENT_FLAG] = True
            ctx = dataclasses.replace(ctx, extra=existing_extra)

            # 9. 实例化 agent_cls 并 build（历史消息由 build 内部管道处理）
            agent_instance = agent_cls()
            subagent = agent_instance.build(ctx)

            # 10. 执行子 Agent
            progress_callback = kwargs.get("progress_callback")
            result_text, events = self._run_subagent(
                subagent,
                timeout_seconds=spec.timeout_seconds,
                execute_kwargs=subagent_execute_kwargs,
                progress_callback=progress_callback,
            )

            # 11. 更新 session 状态为 FINISHED
            try:
                rm.update_session_status(session_code, SessionsStatus.FINISHED.value)
            except Exception as exc:
                logger.warning("LocalBackend update session status to FINISHED failed: %s", exc, exc_info=True)

            return build_enriched_result(
                status="completed",
                agent_name=spec.name,
                agent_type=spec.backend_type,
                summary=result_text,
                tool_calls=count_tool_calls(events),
                exit_reason=ExitReason.COMPLETED.value,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("LocalBackend.execute failed for %s: %s", spec.name, exc)
            try:
                rm.update_session_status(session_code, SessionsStatus.FAILED.value)
            except Exception:
                logger.warning("LocalBackend update session status to FAILED failed: %s", session_code)
            return build_enriched_result(
                status="failed",
                agent_name=spec.name,
                agent_type=spec.backend_type,
                error=str(exc),
                exit_reason=ExitReason.BACKEND_ERROR.value,
            )

    # ==================== 内部方法 ====================

    def _extract_execute_kwargs(self, config: Any, *, session_code: str = "", caller_bk_app_code: str = "") -> Any:
        """从 config 中提取主 Agent 的 execute_kwargs 并构造子 Agent 的 execute_kwargs。

        ExecuteKwargs 是 Pydantic model，复制并覆盖关键字段（stream、persist_input、
        session_code、caller_bk_app_code、invoke_timeout）。

        Args:
            config: 运行时配置，可包含 configurable.execute_kwargs
            session_code: 子 Agent 的会话 code
            caller_bk_app_code: 调用方 app_code（从 kwargs 获取）

        Returns:
            构造好的 ExecuteKwargs 对象
        """
        base = ExecuteKwargs()
        # 从 config 获取主 Agent 的 execute_kwargs
        if config and isinstance(config, dict):
            ek = config.get("configurable", {}).get("execute_kwargs")
            if ek and isinstance(ek, ExecuteKwargs):
                base = ek.model_copy()

        base.stream = True
        base.persist_input = True
        base.session_code = session_code
        if caller_bk_app_code:
            base.caller_bk_app_code = caller_bk_app_code

        return base

    def _run_subagent(
        self,
        subagent: Any,
        *,
        timeout_seconds: int,
        execute_kwargs: Any = None,
        progress_callback: Any = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """流式执行子 Agent，收集文本结果和完整事件列表（供 api_calls 统计）。

        execute_kwargs 由 execute() 中 _extract_execute_kwargs 处理好传入，
        已包含 stream=True、persist_input=True、session_code、invoke_timeout 等字段。

        Args:
            subagent: 已 build 的子 Agent（messages 已含当前用户消息）
            timeout_seconds: 超时秒数
            execute_kwargs: 由 execute() 处理好的 ExecuteKwargs
            progress_callback: 心跳回调

        Returns:
            (result_text, events_list)
        """
        # execute_kwargs 若未传入，构造默认 ExecuteKwargs
        if execute_kwargs is None:
            from aidev_agent.pydantic_models import ExecuteKwargs

            execute_kwargs = ExecuteKwargs(stream=True, invoke_timeout=timeout_seconds)
        else:
            if hasattr(execute_kwargs, "invoke_timeout") and execute_kwargs.invoke_timeout is None:
                execute_kwargs.invoke_timeout = timeout_seconds

        result = subagent.execute(execute_kwargs)

        # 兼容非流式返回（mock 可能返回 dict）
        if not hasattr(result, "__iter__") or isinstance(result, (dict, str)):
            return self._extract_text(result), []

        text_parts: list[str] = []
        events: list[dict[str, Any]] = []

        try:
            for line in result:
                event = self._parse_sse(line)
                if event is None:
                    continue
                events.append(event)
                if event.get("type") == "TEXT_MESSAGE_START":
                    text_parts.clear()
                elif event.get("type") == "TEXT_MESSAGE_CONTENT":
                    delta = event.get("delta", "")
                    if isinstance(delta, str):
                        text_parts.append(delta)

                if progress_callback:
                    step_content = detect_intermediate_step(event, events)
                    if step_content:
                        progress_callback("subagent.intermediate_steps", content=step_content)
                    progress_callback(
                        "subagent.heartbeat",
                        tool_count=count_tool_calls(events),
                        iteration=len(events),
                    )

            return "".join(text_parts), events
        except TimeoutError:
            subagent_name = getattr(subagent, "name", None) or getattr(subagent, "agent_code", None) or "unknown"
            logger.warning(
                "A2A subagent timeout | agent_name=%s timeout_seconds=%s backend_type=local",
                subagent_name,
                timeout_seconds,
            )
            raise

    @staticmethod
    def _parse_sse(line: str) -> dict[str, Any] | None:
        """解析 SSE 编码行 "data: <json>" 为 dict。

        Args:
            line: SSE 编码行，格式为 "data: <json>" 或 "data: [DONE]"

        Returns:
            解析后的 dict，无法解析时返回 None
        """
        if not line.startswith("data: "):
            return None
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            return None
        try:
            return json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _extract_text(result: Any) -> str:
        """从 ``subagent.execute`` 的非流式返回中提取文本。

        非流式返回格式（见 ``ChatCompletionAgent._execute`` chat.py:168-175）：
        ``{"choices": [{"delta": {"role": "assistant", "content": <text>}}], ...}``

        兼容 ``str`` 返回（旧协议）和未识别格式（返回空字符串）。

        Args:
            result: ``subagent.execute`` 的返回值

        Returns:
            提取出的文本，无法识别时返回空字符串
        """
        if isinstance(result, dict):
            choices = result.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str):
                    return content
        if isinstance(result, str):
            return result
        return ""
