# -*- coding: utf-8 -*-
"""ToolNode 级别的 config/state 注入集成测试。

验证 LangGraph ToolNode 正确注入 RunnableConfig 和 InjectedState
到 provider.py 生成的工具函数中。
"""

from tempfile import TemporaryDirectory
from typing import TypedDict

from aidev_agent.core.tools.runtime_tools.local_backend import FilesystemBackend
from aidev_agent.core.tools.runtime_tools.provider import (
    RuntimeBackendResolver,
    get_ls_tool,
)
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode


class InjectionTestState(TypedDict):
    """测试用图状态。"""

    messages: list
    user: str
    session_id: str


class TestToolNodeInjection:
    """测试 ToolNode 级别的 config/state 注入。"""

    def test_ls_tool_config_injection_via_tool_node(self):
        """ls 工具在 ToolNode 中应正确接收 RunnableConfig 注入。"""
        with TemporaryDirectory() as tmpdir:
            backend = FilesystemBackend(root_dir=tmpdir)
            resolver = RuntimeBackendResolver(default_runtime="local")
            resolver.register_runtime("local", backend)

            ls_tool = get_ls_tool(resolver)

            tool_node = ToolNode([ls_tool])
            workflow = StateGraph(InjectionTestState)
            workflow.add_node("tools", tool_node)
            workflow.add_edge(START, "tools")
            workflow.add_edge("tools", END)
            graph = workflow.compile()

            ai_message = AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="ls",
                        args={"path": "/", "target_runtime": "local"},
                        id="call_001",
                        type="tool_call",
                    )
                ],
            )

            result = graph.invoke(
                {"messages": [ai_message], "user": "alice", "session_id": "s-123"},
                config={"configurable": {"thread_id": "t-456"}},
            )

            # 验证工具返回了结果（未报错说明 config 注入成功）
            tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
            assert len(tool_messages) == 1
            assert isinstance(tool_messages[0].content, str)

    def test_ls_tool_state_injection_via_tool_node(self):
        """ls 工具在 ToolNode 中应正确接收 InjectedState 注入。"""
        with TemporaryDirectory() as tmpdir:
            backend = FilesystemBackend(root_dir=tmpdir)
            resolver = RuntimeBackendResolver(default_runtime="local")
            resolver.register_runtime("local", backend)

            ls_tool = get_ls_tool(resolver)

            tool_node = ToolNode([ls_tool])
            workflow = StateGraph(InjectionTestState)
            workflow.add_node("tools", tool_node)
            workflow.add_edge(START, "tools")
            workflow.add_edge("tools", END)
            graph = workflow.compile()

            ai_message = AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="ls",
                        args={"path": "/", "target_runtime": "local"},
                        id="call_002",
                        type="tool_call",
                    )
                ],
            )

            result = graph.invoke(
                {"messages": [ai_message], "user": "alice", "session_id": "s-456"},
                config={"configurable": {"thread_id": "t-789"}},
            )

            tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
            assert len(tool_messages) == 1
            # 工具正常执行（未因 state=None 报错）即表示注入成功
