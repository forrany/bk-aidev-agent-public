from ag_ui.core.events import ThinkingEndEvent, ToolCallResultEvent, ToolCallStartEvent
from pydantic import Field


class ExtendToolCallStartEvent(ToolCallStartEvent):
    description: str | None = None
    mcp_name: str | None = Field(default=None, description="工具调用的模型名称")


class ExtendToolCallResultEvent(ToolCallResultEvent):
    duration: float | None = Field(default=None, description="工具调用的耗时")


class ExtendThinkingEndEvent(ThinkingEndEvent):
    """扩展 ThinkingEndEvent, 添加 duration 字段"""

    duration: float | None = Field(default=None, description="推理过程的耗时")
