# -*- coding: utf-8 -*-
"""A2A Agent 工具子类 — 在 run()/arun() 中将 ToolMessage 包装为 Command 更新 bk_agent_team state。

A2AAgentTool 子类化 StructuredTool，重写 run()/arun()：
1. 调用 super().run() 获取 ToolMessage（BaseTool.run() 自动将 str → ToolMessage）
2. 解析 ToolMessage.content JSON 提取 member/task 信息
3. 构造 Command(update={"bk_agent_team": {...}, "messages": [ToolMessage]}) 返回

Command 构造逻辑完全在 A2AAgentTool 子类中
"""

from __future__ import annotations

import json
from logging import getLogger
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from aidev_agent.core.tools.a2a_tools.types import (
    KEY_AGENT_NAME,
    KEY_EXIT_REASON,
    KEY_MEMBER_NAME,
    KEY_SESSION_CODE,
    KEY_STATUS,
    KEY_TOOL_CALLS,
)

_logger = getLogger(__name__)


def _try_extract_member_info(
    tool_args: dict,
    msg: ToolMessage,
) -> tuple[str, str, str] | None:
    """尝试从 Agent 工具调用的请求参数和返回结果中提取 member 信息。
    关键变化：不再检查工具名（A2AAgentTool 本身就是 Agent 工具子类，无需运行时判断），
    签名从 tool_call: dict 改为 tool_args: dict（直接接收 run() 的 tool_input）。

    仅当 mode="member" 且返回结果包含 session_code 时才提取。

    Args:
        tool_args: 工具调用参数字典，包含 mode, agent_name, member_name 等。
        msg: super().run() 返回的 ToolMessage。

    Returns:
        (member_name, session_code, agent_name) 元组，不符合条件时返回 None。
    """
    mode = tool_args.get("mode")
    if mode != "member":
        return None

    agent_name = tool_args.get("agent_name", "")
    if not agent_name:
        return None

    member_name = tool_args.get("member_name") or agent_name

    content = msg.content if isinstance(msg.content, str) else ""
    if not content:
        return None

    try:
        result = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(result, dict):
        return None

    session_code = result.get(KEY_SESSION_CODE, "")
    if not session_code:
        return None

    effective_member_name = result.get(KEY_MEMBER_NAME, member_name)
    effective_agent_name = result.get(KEY_AGENT_NAME, agent_name)

    return effective_member_name, session_code, effective_agent_name


def _try_extract_task_lifecycle(
    tool_args: dict,
    msg: ToolMessage,
    tool_call_id: str,
) -> tuple[str, dict[str, Any]] | None:
    """从 task 模式 Agent 工具结果中提取生命周期信息。

    迁移自 team_wrapper._try_extract_task_lifecycle（Phase 28, D-05, D-06）。
    关键变化：不再检查工具名；签名从 tool_call: dict 改为 tool_args + tool_call_id 分离参数。

    使用 f"{agent_name}:{tool_call_id}" 作为状态 key（Phase 25 决策，D-06）。

    Args:
        tool_args: 工具调用参数字典。
        msg: super().run() 返回的 ToolMessage。
        tool_call_id: 工具调用 ID，从 run() 的关键字参数获取。

    Returns:
        用于 bk_agent_team 更新的 (key, entry) 元组，不适用时返回 None。
    """
    mode = tool_args.get("mode")
    if mode == "member":
        return None

    agent_name = tool_args.get("agent_name", "")
    if not agent_name:
        return None

    key = f"{agent_name}:{tool_call_id}"

    content = msg.content if isinstance(msg.content, str) else ""
    if not content:
        return None

    try:
        result = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(result, dict):
        return None

    entry: dict[str, Any] = {
        KEY_AGENT_NAME: agent_name,
        KEY_STATUS: result.get(KEY_STATUS, "unknown"),
        "tool_call_id": tool_call_id,
    }
    for field in (KEY_EXIT_REASON, KEY_TOOL_CALLS):
        if field in result:
            entry[field] = result[field]

    return key, entry


class A2AAgentTool(StructuredTool):
    """Agent 工具子类：在 run()/arun() 中将 ToolMessage 包装为 Command 更新 bk_agent_team state。

    替代 team_wrapper.py 在 ToolNode wrapper 链中的拦截逻辑（Phase 28）。

    工具函数（agent_call）保持纯 str 返回（D-02），Command 构造逻辑完全在此子类中。
    A2AAgentTool.from_function() 使用 cls() 构造（继承自 StructuredTool.from_function），
    因此 A2AAgentTool.from_function(func=agent_call, ...) 自动创建 A2AAgentTool 实例。
    """

    def run(
        self,
        tool_input: str | dict,
        *,
        tool_call_id: str | None = None,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> Any:
        """重写 run()：调用 super().run() 获取 ToolMessage，解析后返回 Command。

        super().run() 调用链：invoke() → run() → _run() → func() → str → _format_output() → ToolMessage。
        由于 Command 继承 ToolOutputMixin，若我们返回 Command，_format_output() 不会二次包装。

        Args:
            tool_input: 工具输入参数（dict when from ToolCall, str otherwise）。
            tool_call_id: 工具调用 ID，由 BaseTool._prep_run_args() 从 ToolCall input 提取。
            config: 运行时配置。
            **kwargs: 其他关键字参数。
        """
        result = super().run(tool_input, tool_call_id=tool_call_id, config=config, **kwargs)

        if not isinstance(result, ToolMessage):
            return result

        tool_args = tool_input if isinstance(tool_input, dict) else {}

        # 尝试 member 模式提取（D-05 等价迁移）
        member_info = _try_extract_member_info(tool_args, result)
        if member_info is not None:
            member_name, session_code, agent_name = member_info
            _logger.debug("A2AAgentTool: 将 Agent member ToolMessage 转为 Command for %s", member_name)
            return Command(
                update={
                    "bk_agent_team": {
                        member_name: {
                            KEY_SESSION_CODE: session_code,
                            KEY_STATUS: "active",
                            KEY_AGENT_NAME: agent_name,
                        }
                    },
                    "messages": [result],
                },
            )

        # 尝试 task 模式生命周期提取（D-05/D-06 等价迁移）
        task_info = _try_extract_task_lifecycle(tool_args, result, tool_call_id or "unknown")
        if task_info is not None:
            key, entry = task_info
            _logger.debug("A2AAgentTool: 注册 task 模式生命周期 for %s", key)
            return Command(
                update={
                    "bk_agent_team": {key: entry},
                    "messages": [result],
                },
            )

        return result

    async def arun(
        self,
        tool_input: str | dict,
        *,
        tool_call_id: str | None = None,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> Any:
        """重写 arun()：异步路径，逻辑与 run() 完全一致（D-08）。

        当前 agent_call 无 coroutine，ainvoke() 通过 run_in_executor 委托给 invoke() → run()，
        因此仅重写 run() 即可覆盖异步路径。但按 D-08 要求同时重写 arun() 以防将来添加 coroutine。

        注意：arun() 的签名与 run() 不同（BaseTool.arun 有额外的 verbose/color 等参数），
        但 StructuredTool.arun() 的签名与 run() 一致，因此直接调用 self.run() 即可。
        """
        return self.run(tool_input, tool_call_id=tool_call_id, config=config, **kwargs)
