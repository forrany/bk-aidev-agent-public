# -*- coding: utf-8 -*-
"""ContextProcessor 中间件集成测试（不依赖真实 LLM）。"""

from aidev_agent.core.nodes.model import ContextProcessor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool


@tool
def tool_a() -> str:
    """tool a"""

    return "a"


@tool
def tool_b() -> str:
    """tool b"""

    return "b"


class TestContextProcessorIntegration:
    def test_custom_tool_middleware_can_filter_tools(self):
        cp = ContextProcessor(
            use_structured_response=False,
            enable_query_clarification=True,
            rejection_message="rej",
            role_prompt="role",
            use_general_knowledge_on_miss=False,
            tools=[tool_a, tool_b],
        )

        def filter_mw(ctx, next_):
            next_()
            ctx.tools = [t for t in ctx.tools if t.name == "tool_b"]

        cp.add_tool_middleware(filter_mw)

        tools = cp.get_choice_tools(state={}, config={})
        assert [t.name for t in tools] == ["tool_b"]

    def test_custom_variable_middleware_can_inject_variable(self):
        cp = ContextProcessor(
            use_structured_response=False,
            enable_query_clarification=True,
            rejection_message="rej",
            role_prompt="role",
            use_general_knowledge_on_miss=False,
            tools=[],
        )

        def inject_mw(ctx, next_):
            next_()
            ctx.variables["foo"] = "bar"

        cp.add_variable_middleware(inject_mw)

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        variables = cp.get_chat_prompt_variables(
            chat_prompt_template=prompt,
            state={"input": "hi", "messages": []},
            config={},
        )

        assert variables["foo"] == "bar"
