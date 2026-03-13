# -*- coding: utf-8 -*-
"""BKAI 远程后端：通过 API 网关直调子智能体。

调用模式：
- execute(task): 带 session_code，一次性任务模式
- execute(member): 带 session_code，多轮会话模式

execute() 统一处理 task 和 member 两种调用模式。

Client 注入设计：
- Client 在 spec 构造时创建并注入到 spec.params["client"]（由 ChatAgentBuilder.build_subagents 注入）
- BkaiBackend 从 spec.params["client"] 获取 Client，不负责构造
- 用户可注入自定义 client（其他 bkai 鉴权方式），无需修改 BkaiBackend

鉴权设计：
- Client 通过 X-Bkapi-Authorization header 携带用户态 access_token
- 鉴权由 Client 构造方在创建时统一注入
- BkaiBackend 无需手动设置鉴权 header
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from logging import getLogger
from typing import Any

from aidev_agent.config import settings
from aidev_agent.core.tools.a2a_tools.progress import build_enriched_result, detect_intermediate_step
from aidev_agent.core.tools.a2a_tools.types import AgentResult, AgentSpec, ExitReason
from aidev_agent.pydantic_models import ExecuteKwargs

logger = getLogger(__name__)


class BkaiBackend:
    """BKAI 远程后端：通过 API 网关直调子智能体。

    通过 execute() 统一处理 task 和 member 两种调用模式（session_code 总是非空）。
    Phase 23: execute() 改为流式执行，收集 SSE 事件统计 api_calls。
    Phase 29: session_code 总是非空（由 provider 生成），移除所有 if session_code 条件分支；
    execute_kwargs 从 config.configurable.execute_kwargs 获取并修改关键字段。
    零参构造（D-07），运行时依赖通过 execute kwargs 注入（D-08）。
    Client 从 spec.params["client"] 获取，不负责构造，无 fallback。

    鉴权由 Client session 的 X-Bkapi-Authorization header 统一处理，
    BkaiBackend 无需手动设置鉴权相关 header。
    """

    def execute(self, spec: AgentSpec, message: str, *, session_code: str = "", **kwargs: Any) -> AgentResult:
        """通过流式请求调用子智能体（task / member 统一入口 + 可观测性）。

        Args:
            spec: Agent 规格，params 中需包含 client
            message: 发送给子 Agent 的消息
            session_code: 会话 code；必须非空
            **kwargs: 运行时依赖（config 用于提取 execute_kwargs）

        Returns:
            AgentResult 标准化富结果（不可变 Pydantic 模型）

        Raises:
            ValueError: session_code 为空或 client 缺失时
        """
        # 1. session_code 必须非空
        if not session_code:
            raise ValueError("session_code must not be empty for BkaiBackend")
        # 2. 内联 _get_client 逻辑
        client = spec.params.get("client")
        if not client:
            raise ValueError(f"Missing 'client' in spec.params for agent {spec.name}")

        # 3. 获取 executor（优先从 execute_kwargs 获取）
        caller_bk_app_code = spec.params.get("caller_bk_app_code", "")
        config = kwargs.get("config")
        execute_kwargs_dict = self._extract_execute_kwargs(
            config, session_code=session_code, caller_bk_app_code=caller_bk_app_code
        )

        # 4. 获取 executor
        executor = execute_kwargs_dict.get("executor", "")

        # 5. 条件执行 prepare_session
        should_prepare_session = spec.params.get("should_prepare_session", False)
        if should_prepare_session:
            self.prepare_session(client, session_code, f"sa-{session_code[:8]}", message, executor=executor)

        # 6. 获取 should_openapi_chat_completion
        should_openapi = spec.params.get("should_openapi_chat_completion", False)

        # 7. 流式执行
        try:
            text_parts: list[str] = []
            events: list[dict[str, Any]] = []
            progress_callback = kwargs.pop("progress_callback", None)

            for event in self._chat_completion_stream_via_client(
                client,
                spec,
                message,
                session_code=session_code,
                base_execute_kwargs=execute_kwargs_dict,
                should_openapi_chat_completion=should_openapi,
                progress_callback=progress_callback,
                **kwargs,
            ):
                if not isinstance(event, dict):
                    continue
                events.append(event)
                if event.get("type") == "TEXT_MESSAGE_START":
                    text_parts.clear()
                elif event.get("type") == "TEXT_MESSAGE_CONTENT":
                    delta = event.get("delta", "")
                    if isinstance(delta, str):
                        text_parts.append(delta)
                elif event.get("type") == "error":
                    raise RuntimeError(event.get("error", "Unknown stream error"))

            result_text = "".join(text_parts)
            api_calls = sum(1 for e in events if e.get("type") == "TOOL_CALL_START")
            return build_enriched_result(
                status="completed",
                agent_name=spec.name,
                agent_type=spec.backend_type,
                summary=result_text,
                tool_calls=api_calls,
                exit_reason=ExitReason.COMPLETED.value,
            )
        except Exception as exc:
            exit_reason = ExitReason.BACKEND_ERROR.value
            error_msg = str(exc)
            if "401" in error_msg or "credential" in error_msg.lower():
                exit_reason = ExitReason.CREDENTIAL_ERROR.value
            elif "timeout" in error_msg.lower():
                exit_reason = ExitReason.TIMEOUT.value
                logger.warning(
                    "A2A subagent timeout | agent_name=%s timeout_seconds=%s backend_type=bkai session_code=%s",
                    spec.name,
                    spec.timeout_seconds,
                    session_code,
                )
            logger.warning("BkaiBackend.execute failed for %s: %s", spec.name, exc)
            return build_enriched_result(
                status="failed",
                agent_name=spec.name,
                agent_type=spec.backend_type,
                error=error_msg,
                exit_reason=exit_reason,
            )

    def prepare_session(
        self,
        client: Any,
        session_code: str,
        session_name: str,
        message: str,
        *,
        executor: str = "",
    ) -> None:
        """准备远端子智能体的 session：创建 session + 保存 user 消息。

        与 LocalBackend.prepare_session 类似，但通过远端 client API 操作（非 resource_manager）。
        BkaiBackend 不负责保存 assistant 消息（远端子 Agent 自行管理）。
        也不负责加载历史上下文（远端子 Agent 自行管理）。

        Args:
            client: 已构造好的 Client 实例（通过 spec.params["client"] 获取）
            session_code: 预生成的 session code（UUID hex）
            session_name: session 显示名称
            message: 用户输入消息
            executor: 执行人用户名
        """
        # 创建远端 session（幂等：若 session 已存在，后端返回 200）
        try:
            res = client.create_session(
                data={
                    "is_temporary": settings.BKAI_A2A_SESSION_TEMPORARY,
                    "session_code": session_code,
                    "session_name": session_name,
                },
                headers={"X-BKAIDEV-USER": executor},
            )
            logger.info("BkaiBackend prepare_session create_session success: %s", res)
        except Exception as exc:
            logger.warning("BkaiBackend prepare_session create_session failed: %s", exc, exc_info=True)

        # 保存 user 消息到远端 session_content
        try:
            client.save_session_content(data={"session_code": session_code, "role": "user", "content": message})
        except Exception as exc:
            logger.warning("BkaiBackend prepare_session save_session_content failed: %s", exc, exc_info=True)

    # ---------- 内部方法 ----------

    def _extract_execute_kwargs(
        self, config: Any, *, session_code: str = "", stream: bool = True, caller_bk_app_code: str = ""
    ) -> dict[str, Any]:
        """从 config 中提取主 Agent 的 execute_kwargs 并构造子 Agent 的 execute_kwargs。

        ExecuteKwargs 是 Pydantic model，直接用 model_dump() 序列化。
        覆盖关键字段（session_code、stream、persist_input），
        并将主 Agent 的 app_code 写入 caller_bk_app_code。

        Args:
            config: 运行时配置，可包含 configurable.execute_kwargs
            session_code: 子 Agent 的会话 code
            stream: 是否启用流式模式，默认 True
            caller_bk_app_code: 调用方 app_code（从 spec.params 获取）

        Returns:
            构造好的 execute_kwargs 字典
        """
        base_execute_kwargs: dict[str, Any] = {}
        if config and isinstance(config, dict):
            ek = config.get("configurable", {}).get("execute_kwargs")
            if ek and isinstance(ek, ExecuteKwargs):
                base_execute_kwargs = ek.model_dump()

        base_execute_kwargs["stream"] = stream
        base_execute_kwargs["persist_input"] = True
        base_execute_kwargs["session_code"] = session_code

        if caller_bk_app_code:
            base_execute_kwargs["caller_bk_app_code"] = caller_bk_app_code

        return base_execute_kwargs

    def _handle_request_error(self, exc: Exception, context: str, spec: AgentSpec) -> dict[str, Any]:
        """统一异常转错误字典。

        Args:
            exc: 捕获的异常
            context: 错误上下文描述（如 "chat_completion"）
            spec: Agent 规格（用于 agent_name）

        Returns:
            标准错误结果字典
        """
        logger.warning("BkaiBackend %s failed for %s: %s", context, spec.name, exc, exc_info=True)
        return {"status": "failed", "agent_name": spec.name, "error": str(exc)}

    def _chat_completion_stream_via_client(
        self,
        client: Any,
        spec: AgentSpec,
        message: str,
        *,
        session_code: str = "",
        base_execute_kwargs: dict[str, Any] | None = None,
        should_openapi_chat_completion: bool = False,
        progress_callback: Any = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """通过 Client 的 API 操作发送流式请求。

        优先使用 operation.request 方法发送流式请求，返回 requests.Response 后
        迭代 SSE 事件。根据 should_openapi_chat_completion 选择 private 或 openapi 端点。

        Args:
            client: 已构造好的 Client 实例
            spec: Agent 规格
            message: 发送给子 Agent 的消息
            session_code: 会话 code
            base_execute_kwargs: 已构造的 execute_kwargs 字典（由 execute 传入）
            should_openapi_chat_completion: 使用 openapi 端点
            progress_callback: 可选，用于发送 subagent.heartbeat 事件
            **kwargs: 运行时依赖

        Yields:
            解析后的 SSE 事件字典
        """
        if base_execute_kwargs is None:
            base_execute_kwargs = {}

        payload: dict[str, Any] = {
            "input": message,
            "execute_kwargs": base_execute_kwargs,
            "session_code": session_code,
            "session_temporary": settings.BKAI_A2A_SESSION_TEMPORARY,
        }

        # 根据 should_openapi_chat_completion 选择对应的 Operation
        chat_op = client.openapi_chat_completion if should_openapi_chat_completion else client.private_chat_completion

        try:
            # 优先使用 operation.request 发送流式请求
            try:
                resp = chat_op.request(
                    json=payload,
                    timeout=spec.timeout_seconds,
                    verify=False,
                    stream=True,
                )
            except (TypeError, AttributeError):
                # 降级：使用 client.session.post 发送流式请求
                url = f"{client._endpoint.rstrip('/')}{chat_op.path}"  # noqa: SLF001
                headers = dict(client.session.headers)
                resp = client.session.post(
                    url, json=payload, headers=headers, verify=False, timeout=spec.timeout_seconds, stream=True
                )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("BkaiBackend chat_completion_with_stream HTTP error: %s", exc)
            yield {"type": "error", "error": str(exc)}
            return

        start_time = time.time()
        events: list[dict[str, Any]] = []

        for raw_line in resp.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                    events.append(event)
                    # 通过 progress_callback 发送中间步骤和心跳
                    if progress_callback:
                        step_content = detect_intermediate_step(event, events)
                        if step_content:
                            progress_callback("subagent.intermediate_steps", content=step_content)
                        elapsed = time.time() - start_time
                        tool_count = sum(1 for e in events if e.get("type") == "TOOL_CALL_START")
                        progress_callback(
                            "subagent.heartbeat",
                            elapsed_seconds=round(elapsed, 3),
                            tool_count=tool_count,
                            iteration=len(events),
                        )
                    yield event
                except (json.JSONDecodeError, ValueError):
                    # 非 JSON 行（如 STATE_SNAPSHOT 跨行）跳过
                    continue
