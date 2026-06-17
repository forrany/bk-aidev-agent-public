# -*- coding: utf-8 -*-
"""A2A 工具模块。

提供 Agent-to-Agent 调用能力，支持 Task（一次性任务）和 Member（多轮对话）两种模式。
Phase 23: 增加可观测性类型导出（ExitReason、AgentResult、progress 工具函数）。
"""

from aidev_agent.core.tools.a2a_tools.agent_tool import A2AAgentTool
from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend
from aidev_agent.core.tools.a2a_tools.local_backend import LocalBackend
from aidev_agent.core.tools.a2a_tools.progress import build_enriched_result, count_tool_calls, sanitize_error_message
from aidev_agent.core.tools.a2a_tools.provider import get_agent_tools
from aidev_agent.core.tools.a2a_tools.types import (
    AgentBackend,
    AgentBackendType,
    AgentResult,
    AgentSpec,
    AgentToolInput,
    ExitReason,
    ProgressCallback,
    SendMessageInput,
    SubAgentConfig,
)

__all__ = [
    "A2AAgentTool",
    "AgentBackend",
    "AgentBackendType",
    "AgentResult",
    "AgentSpec",
    "AgentToolInput",
    "BkaiBackend",
    "ExitReason",
    "LocalBackend",
    "ProgressCallback",
    "SendMessageInput",
    "SubAgentConfig",
    "build_enriched_result",
    "count_tool_calls",
    "get_agent_tools",
    "sanitize_error_message",
]
