# -*- coding: utf-8 -*-
"""测试 ContextProcessor 工具选择中间件。"""

from aidev_agent.core.nodes.model.basic_middleware import BaseToolsMiddleware, DecisionBasedToolFilterMiddleware
from aidev_agent.core.nodes.model.context_processor import MiddlewarePipeline
from aidev_agent.core.nodes.model.pydantic_models import ProcessorContext
from aidev_agent.enums import Decision
from langchain_core.tools import tool


@tool
def tool_a() -> str:
    """tool a"""

    return "a"


@tool
def tool_b() -> str:
    """tool b"""

    return "b"


class TestToolsMiddleware:
    def test_base_tools_middleware_sets_all_tools(self):
        pipeline = MiddlewarePipeline()
        pipeline.use(BaseToolsMiddleware())

        ctx = ProcessorContext(state={}, config={}, metadata={"all_tools": [tool_a, tool_b]})
        pipeline.execute(ctx)

        assert [t.name for t in ctx.tools] == ["tool_a", "tool_b"]

    def test_decision_based_filter_applies_allowlist(self):
        pipeline = MiddlewarePipeline()
        pipeline.use(BaseToolsMiddleware())
        pipeline.use(DecisionBasedToolFilterMiddleware())

        ctx = ProcessorContext(
            state={"decision": Decision.GENERAL_QA},
            config={},
            metadata={
                "all_tools": [tool_a, tool_b],
                "tools_allowlist_by_decision": {Decision.GENERAL_QA: {"tool_b"}},
            },
        )
        pipeline.execute(ctx)

        assert [t.name for t in ctx.tools] == ["tool_b"]
