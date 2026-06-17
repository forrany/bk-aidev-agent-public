# -*- coding: utf-8 -*-
"""A2A 可观测性工具函数。

提供流式事件解析、错误脱敏和富结果构建。tools 层内部使用。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from aidev_agent.core.tools.a2a_tools.types import AgentResult, ExitReason

logger = logging.getLogger("aidev-agent")


def count_tool_calls(events: list[dict[str, Any]]) -> int:
    """从 AGUI 流式事件列表中统计 TOOL_CALL_START 事件次数。

    Args:
        events: AGUI 流式事件列表，每个事件包含 type 字段

    Returns:
        TOOL_CALL_START 事件次数
    """
    return sum(1 for e in events if e.get("type") == "TOOL_CALL_START")


def detect_intermediate_step(event: dict[str, Any], all_events: list[dict[str, Any]]) -> str | None:
    """从 SSE 事件中检测子 Agent 执行步骤变更。

    根据事件类型返回对应的中文描述，用于通过 progress_callback("subagent.intermediate_steps")
    通知前端当前子 Agent 执行状态。

    Args:
        event: 当前 SSE 事件
        all_events: 截至当前的所有 SSE 事件列表（用于计算工具执行次序）

    Returns:
        步骤描述文本，无法识别时返回 None
    """
    event_type = event.get("type", "")

    if event_type == "TOOL_CALL_START":
        tool_name = event.get("toolCallName", "unknown")
        tool_count = sum(1 for e in all_events if e.get("type") == "TOOL_CALL_START")
        return f"正在执行工具 (第{tool_count}次, {tool_name})"

    if event_type == "TEXT_MESSAGE_START":
        return "正在进行模型输出"

    if event_type == "CUSTOM":
        name = event.get("name", "")
        if name.startswith("knowledge_rag"):
            return "正在进行知识库召回"

    return None


def sanitize_error_message(message: str) -> str:
    """过滤错误消息中的敏感信息。

    移除以下模式：
    - API key (sk- 前缀长字符串)
    - access_token 参数值
    - Bearer token

    Args:
        message: 原始错误消息

    Returns:
        脱敏后的错误消息
    """
    sanitized = message
    # 移除 API key 模式 (sk- 前缀)
    sanitized = re.sub(r"sk-[a-zA-Z0-9]{20,}", "[REDACTED_API_KEY]", sanitized)
    # 移除 access_token 参数值
    sanitized = re.sub(r'access_token[=:]\s*["\']?[a-zA-Z0-9_-]{10,}["\']?', "access_token=[REDACTED]", sanitized)
    # 移除 Bearer token
    sanitized = re.sub(r"Bearer\s+[a-zA-Z0-9_\-.]{20,}", "Bearer [REDACTED]", sanitized)
    return sanitized


def build_enriched_result(
    *,
    status: str,
    agent_name: str = "",
    agent_type: str = "",
    summary: str | None = None,
    error: str | None = None,
    tool_calls: int = 0,
    exit_reason: str = ExitReason.COMPLETED.value,
) -> AgentResult:
    """构建标准化富结果。

    Phase 26：返回 AgentResult(BaseModel) 替代 dict。移除 duration_seconds / session_code / member_name，
    这些属于 provider 层概念。duration_seconds 由 timer_wrapper 在 ToolNode 层统一记录。

    Args:
        status: 执行状态 ("completed" | "failed")
        agent_name: Agent 名称
        agent_type: 智能体后端类型（如 bkai / local）
        summary: 完成时的文本结果
        error: 失败时的错误信息（会自动脱敏）
        tool_calls: 工具调用次数
        exit_reason: 退出原因（ExitReason 枚举值）

    Returns:
        AgentResult 标准化富结果
    """
    if error:
        error = sanitize_error_message(error)

    result = AgentResult(
        status=status,
        agent_type=agent_type,
        result=summary or "",
        tool_calls=tool_calls,
        exit_reason=exit_reason,  # type: ignore[arg-type]  # str 与 ExitReason 兼容
    )
    if error:
        result = result.model_copy(update={"error": error})
    return result
