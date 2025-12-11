# -*- coding: utf-8 -*-
"""Tool call wrappers: result length limit."""

from collections.abc import Awaitable
from typing import Callable

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import AsyncToolCallWrapper, ToolCallRequest, ToolCallWrapper
from langgraph.types import Command

TOOL_RESULT_TOO_LONG_MESSAGE = "本次工具调用返回结果超长，请重新调整调用参数"


def _tool_msg_content_len(msg: ToolMessage) -> int:
    return len(str(getattr(msg, "content", "")))


def build_result_limit_sync_wrapper(result_limit: int) -> ToolCallWrapper:
    def wrapper(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        msg = execute(request)

        if isinstance(msg, ToolMessage) and _tool_msg_content_len(msg) > result_limit:
            msg.content = TOOL_RESULT_TOO_LONG_MESSAGE

        return msg

    return wrapper


def build_result_limit_async_wrapper(result_limit: int) -> AsyncToolCallWrapper:
    async def wrapper(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        msg = await execute(request)

        if isinstance(msg, ToolMessage) and _tool_msg_content_len(msg) > result_limit:
            msg.content = TOOL_RESULT_TOO_LONG_MESSAGE

        return msg

    return wrapper
