# -*- coding: utf-8 -*-
"""Tool call wrappers: timer."""

import time
from collections.abc import Awaitable
from typing import Callable

from langchain_core.callbacks import dispatch_custom_event
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command


def _add_metadata_to_tool_message(
    msg: ToolMessage,
    duration: int,
    description: str | None,
) -> None:
    msg.additional_kwargs.setdefault("duration", duration)
    if description:
        msg.additional_kwargs.setdefault("description", description)


def timer_sync_wrapper(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    # 使用 time.monotonic() 替代 time.time()，使用 time.time() 计算耗时可能受系统时间调整影响导致负值
    t1 = int(time.monotonic() * 1000)
    msg = execute(request)
    t2 = int(time.monotonic() * 1000)
    duration = t2 - t1

    description = None
    if request.tool:
        description = getattr(request.tool, "description", None)

    if isinstance(msg, ToolMessage):
        _add_metadata_to_tool_message(msg, duration, description)
    elif isinstance(msg, Command) and msg.update:
        for each in msg.update.get("messages", []):
            if isinstance(each, ToolMessage):
                _add_metadata_to_tool_message(each, duration, description)
    dispatch_custom_event("on_tool_node_finish", data=msg, config=request.runtime.config)
    return msg


async def timer_async_wrapper(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
) -> ToolMessage | Command:
    t1 = int(time.monotonic() * 1000)
    msg = await execute(request)
    t2 = int(time.monotonic() * 1000)
    duration = t2 - t1

    description = None
    if request.tool:
        description = getattr(request.tool, "description", None)

    if isinstance(msg, ToolMessage):
        _add_metadata_to_tool_message(msg, duration, description)
    elif isinstance(msg, Command) and msg.update:
        for each in msg.update.get("messages", []):
            if isinstance(each, ToolMessage):
                _add_metadata_to_tool_message(each, duration, description)
    dispatch_custom_event("on_tool_node_finish", data=msg, config=request.runtime.config)
    return msg
