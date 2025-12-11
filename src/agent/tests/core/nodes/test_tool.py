# -*- coding: utf-8 -*-
"""
测试 build_tool_node 功能
"""
import time
from typing import List

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from aidev_agent.core.nodes.tool import build_tool_node


# ============================================================================
# 测试状态定义
# ============================================================================

class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[List, add_messages]


# ============================================================================
# 测试工具定义
# ============================================================================

@tool
def calculator(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiplier(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def failing_tool(message: str) -> str:
    """A tool that always fails."""
    raise ValueError(f"Tool failed: {message}")


@tool
def slow_tool(duration_ms: int = 100) -> str:
    """A tool that takes some time to execute."""
    time.sleep(duration_ms / 1000)
    return f"Slept for {duration_ms}ms"


# ============================================================================
# 辅助函数
# ============================================================================

def run_tool_node_in_graph(tool_node, state: dict) -> dict:
    """在图中运行 tool_node 并返回结果"""
    # 构建一个简单的图来测试 tool_node
    graph = StateGraph(AgentState)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)

    compiled = graph.compile()
    result = compiled.invoke(state)
    return result


async def arun_tool_node_in_graph(tool_node, state: dict) -> dict:
    """在图中异步运行 tool_node 并返回结果"""
    # 构建一个简单的图来测试 tool_node
    graph = StateGraph(AgentState)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)

    compiled = graph.compile()
    result = await compiled.ainvoke(state)
    return result


# ============================================================================
# 测试用例
# ============================================================================

class TestBuildToolNode:
    """测试 build_tool_node 函数"""

    def test_no_tool_calls(self):
        """测试1: 模型返回没有工具调用时的行为"""
        tool_node = build_tool_node(tools=[calculator])

        # 构造状态：没有工具调用
        state = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
            ]
        }

        # 执行 tool_node
        result = run_tool_node_in_graph(tool_node, state)

        # 验证：应该返回空消息列表或不改变状态
        assert isinstance(result, dict)
        assert "messages" in result
        # 没有工具调用，不应生成新的 ToolMessage
        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 0

    def test_single_tool_call_success(self):
        """测试2: 模型返回一个工具调用，工具正常执行"""
        tool_node = build_tool_node(tools=[calculator])

        # 构造状态：包含一个工具调用
        state = {
            "messages": [
                HumanMessage(content="What is 2 + 3?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"a": 2, "b": 3},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ]
                ),
            ]
        }

        # 执行 tool_node
        result = run_tool_node_in_graph(tool_node, state)

        # 验证结果
        assert isinstance(result, dict)
        assert "messages" in result

        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 1

        tool_msg = tool_messages[0]
        assert tool_msg.content == "5"
        assert tool_msg.tool_call_id == "call_1"
        assert tool_msg.name == "calculator"

        # 验证元数据
        assert "duration" in tool_msg.additional_kwargs
        assert isinstance(tool_msg.additional_kwargs["duration"], int)
        assert tool_msg.additional_kwargs["duration"] >= 0

        assert "description" in tool_msg.additional_kwargs
        assert tool_msg.additional_kwargs["description"] == "Add two numbers."

    def test_single_tool_call_with_exception(self):
        """测试3: 模型返回一个工具调用，工具抛出异常"""
        tool_node = build_tool_node(tools=[failing_tool], handle_tool_errors=True)

        # 构造状态：包含一个会失败的工具调用
        state = {
            "messages": [
                HumanMessage(content="Test error handling"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "failing_tool",
                            "args": {"message": "test error"},
                            "id": "call_fail",
                            "type": "tool_call",
                        }
                    ]
                ),
            ]
        }

        # 执行 tool_node，不应抛出异常
        result = run_tool_node_in_graph(tool_node, state)

        # 验证结果
        assert isinstance(result, dict)
        assert "messages" in result

        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 1

        tool_msg = tool_messages[0]
        assert "Tool failed: test error" in tool_msg.content
        assert tool_msg.tool_call_id == "call_fail"
        assert tool_msg.status == "error"

        # 验证元数据（即使失败也应该有执行时长）
        assert "duration" in tool_msg.additional_kwargs
        assert isinstance(tool_msg.additional_kwargs["duration"], int)

    def test_multiple_tool_calls_all_success(self):
        """测试4: 模型返回多个工具调用，所有工具正常执行"""
        tool_node = build_tool_node(tools=[calculator, multiplier])

        # 构造状态：包含两个工具调用
        state = {
            "messages": [
                HumanMessage(content="Calculate 2+3 and 4*5"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"a": 2, "b": 3},
                            "id": "call_1",
                            "type": "tool_call",
                        },
                        {
                            "name": "multiplier",
                            "args": {"a": 4, "b": 5},
                            "id": "call_2",
                            "type": "tool_call",
                        },
                    ]
                ),
            ]
        }

        # 执行 tool_node
        result = run_tool_node_in_graph(tool_node, state)

        # 验证结果
        assert isinstance(result, dict)
        assert "messages" in result

        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 2

        # 验证第一个工具调用
        calc_msg = next(msg for msg in tool_messages if msg.name == "calculator")
        assert calc_msg.content == "5"
        assert calc_msg.tool_call_id == "call_1"
        assert "duration" in calc_msg.additional_kwargs
        assert "description" in calc_msg.additional_kwargs

        # 验证第二个工具调用
        mult_msg = next(msg for msg in tool_messages if msg.name == "multiplier")
        assert mult_msg.content == "20"
        assert mult_msg.tool_call_id == "call_2"
        assert "duration" in mult_msg.additional_kwargs
        assert "description" in mult_msg.additional_kwargs

    def test_multiple_tool_calls_with_failures(self):
        """测试5: 模型返回多个工具调用，部分工具抛出异常"""
        tool_node = build_tool_node(
            tools=[calculator, failing_tool],
            handle_tool_errors=True
        )

        # 构造状态：包含一个成功和一个失败的工具调用
        state = {
            "messages": [
                HumanMessage(content="Test mixed results"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"a": 2, "b": 3},
                            "id": "call_success",
                            "type": "tool_call",
                        },
                        {
                            "name": "failing_tool",
                            "args": {"message": "test"},
                            "id": "call_fail",
                            "type": "tool_call",
                        },
                    ]
                ),
            ]
        }

        # 执行 tool_node
        result = run_tool_node_in_graph(tool_node, state)

        # 验证结果
        assert isinstance(result, dict)
        assert "messages" in result

        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 2

        # 验证成功的工具调用
        success_msg = next(msg for msg in tool_messages if msg.name == "calculator")
        assert success_msg.content == "5"
        assert success_msg.tool_call_id == "call_success"
        assert "duration" in success_msg.additional_kwargs

        # 验证失败的工具调用
        fail_msg = next(msg for msg in tool_messages if msg.name == "failing_tool")
        assert "Tool failed: test" in fail_msg.content
        assert fail_msg.tool_call_id == "call_fail"
        assert fail_msg.status == "error"
        assert "duration" in fail_msg.additional_kwargs

    def test_timing_accuracy(self):
        """测试6: 验证执行时间记录的准确性"""
        tool_node = build_tool_node(tools=[slow_tool])

        # 构造状态：调用慢速工具
        state = {
            "messages": [
                HumanMessage(content="Test timing"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "slow_tool",
                            "args": {"duration_ms": 100},
                            "id": "call_slow",
                            "type": "tool_call",
                        }
                    ]
                ),
            ]
        }

        # 执行 tool_node
        result = run_tool_node_in_graph(tool_node, state)

        # 验证结果
        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 1

        tool_msg = tool_messages[0]
        assert "duration" in tool_msg.additional_kwargs

        # 执行时间应该大约是 100ms (允许一些误差)
        duration = tool_msg.additional_kwargs["duration"]
        assert duration >= 90  # 至少 90ms
        assert duration <= 200  # 最多 200ms (留一些余量)

    def test_handle_tool_errors_false(self):
        """测试7: 当 handle_tool_errors=False 时，异常应该被抛出"""
        tool_node = build_tool_node(
            tools=[failing_tool],
            handle_tool_errors=False
        )

        # 构造状态：包含一个会失败的工具调用
        state = {
            "messages": [
                HumanMessage(content="Test error propagation"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "failing_tool",
                            "args": {"message": "test"},
                            "id": "call_fail",
                            "type": "tool_call",
                        }
                    ]
                ),
            ]
        }

        # 执行 tool_node，应该抛出异常
        with pytest.raises(ValueError, match="Tool failed: test"):
            run_tool_node_in_graph(tool_node, state)

    def test_custom_wrapper(self):
        """测试8: 自定义包装器是否正常工作"""
        # 定义一个自定义包装器，用于记录工具调用
        call_log = []

        def logging_wrapper(request, execute):
            call_log.append(f"before:{request.tool_call['name']}")
            result = execute(request)
            call_log.append(f"after:{request.tool_call['name']}")
            return result

        tool_node = build_tool_node(
            tools=[calculator],
            wrappers=[logging_wrapper]
        )

        # 构造状态
        state = {
            "messages": [
                HumanMessage(content="What is 2 + 3?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"a": 2, "b": 3},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ]
                ),
            ]
        }

        # 执行 tool_node
        result = run_tool_node_in_graph(tool_node, state)

        # 验证自定义包装器被调用
        assert len(call_log) == 2
        assert call_log[0] == "before:calculator"
        assert call_log[1] == "after:calculator"

        # 验证结果仍然正确
        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].content == "5"

        # 验证元数据仍然存在（timer_wrapper 在自定义 wrapper 之前执行）
        assert "duration" in tool_messages[0].additional_kwargs
        assert "description" in tool_messages[0].additional_kwargs

    def test_multiple_custom_wrappers(self):
        """测试8+: 多个自定义包装器的执行顺序"""
        execution_order = []

        def wrapper1(request, execute):
            execution_order.append("wrapper1_start")
            result = execute(request)
            execution_order.append("wrapper1_end")
            return result

        def wrapper2(request, execute):
            execution_order.append("wrapper2_start")
            result = execute(request)
            execution_order.append("wrapper2_end")
            return result

        tool_node = build_tool_node(
            tools=[calculator],
            wrappers=[wrapper1, wrapper2]
        )

        # 构造状态
        state = {
            "messages": [
                HumanMessage(content="Test wrapper order"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"a": 1, "b": 1},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ]
                ),
            ]
        }

        # 执行 tool_node
        result = run_tool_node_in_graph(tool_node, state)

        # 验证执行顺序：timer -> wrapper1 -> wrapper2 -> tool -> wrapper2 -> wrapper1 -> timer
        # (我们只能看到 wrapper1 和 wrapper2 的顺序)
        assert execution_order == [
            "wrapper1_start",
            "wrapper2_start",
            "wrapper2_end",
            "wrapper1_end",
        ]

    def test_wrapper_can_modify_request(self):
        """测试8++: 包装器可以修改请求"""
        def doubling_wrapper(request, execute):
            # 将参数翻倍
            modified_request = request.override(
                tool_call={
                    **request.tool_call,
                    "args": {
                        "a": request.tool_call["args"]["a"] * 2,
                        "b": request.tool_call["args"]["b"] * 2,
                    }
                }
            )
            return execute(modified_request)

        tool_node = build_tool_node(
            tools=[calculator],
            wrappers=[doubling_wrapper]
        )

        # 构造状态：原始请求是 2 + 3
        state = {
            "messages": [
                HumanMessage(content="Test request modification"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"a": 2, "b": 3},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ]
                ),
            ]
        }

        # 执行 tool_node
        result = run_tool_node_in_graph(tool_node, state)

        # 验证结果：应该是 (2*2) + (3*2) = 4 + 6 = 10
        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].content == "10"


# ============================================================================
# 异步测试
# ============================================================================

@pytest.mark.asyncio
class TestBuildToolNodeAsync:
    """测试 build_tool_node 的异步功能"""

    async def test_no_tool_calls_async(self):
        """测试1: 模型返回没有工具调用时的异步行为"""
        tool_node = build_tool_node(tools=[calculator])

        # 构造状态：没有工具调用
        state = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
            ]
        }

        # 异步执行 tool_node
        result = await arun_tool_node_in_graph(tool_node, state)

        # 验证：应该返回空消息列表或不改变状态
        assert isinstance(result, dict)
        assert "messages" in result
        # 没有工具调用，不应生成新的 ToolMessage
        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 0

    async def test_single_tool_call_success_async(self):
        """测试2: 模型返回一个工具调用，工具正常执行（异步）"""
        tool_node = build_tool_node(tools=[calculator])

        # 构造状态：包含一个工具调用
        state = {
            "messages": [
                HumanMessage(content="What is 2 + 3?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"a": 2, "b": 3},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ]
                ),
            ]
        }

        # 异步执行 tool_node
        result = await arun_tool_node_in_graph(tool_node, state)

        # 验证结果
        assert isinstance(result, dict)
        assert "messages" in result

        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 1

        tool_msg = tool_messages[0]
        assert tool_msg.content == "5"
        assert tool_msg.tool_call_id == "call_1"
        assert tool_msg.name == "calculator"

        # 验证元数据
        assert "duration" in tool_msg.additional_kwargs
        assert isinstance(tool_msg.additional_kwargs["duration"], int)
        assert tool_msg.additional_kwargs["duration"] >= 0

        assert "description" in tool_msg.additional_kwargs
        assert tool_msg.additional_kwargs["description"] == "Add two numbers."

    async def test_single_tool_call_with_exception_async(self):
        """测试3: 模型返回一个工具调用，工具抛出异常（异步）"""
        tool_node = build_tool_node(tools=[failing_tool], handle_tool_errors=True)

        # 构造状态：包含一个会失败的工具调用
        state = {
            "messages": [
                HumanMessage(content="Test error handling"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "failing_tool",
                            "args": {"message": "test error"},
                            "id": "call_fail",
                            "type": "tool_call",
                        }
                    ]
                ),
            ]
        }

        # 异步执行 tool_node，不应抛出异常
        result = await arun_tool_node_in_graph(tool_node, state)

        # 验证结果
        assert isinstance(result, dict)
        assert "messages" in result

        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 1

        tool_msg = tool_messages[0]
        assert "Tool failed: test error" in tool_msg.content
        assert tool_msg.tool_call_id == "call_fail"
        assert tool_msg.status == "error"

        # 验证元数据（即使失败也应该有执行时长）
        assert "duration" in tool_msg.additional_kwargs
        assert isinstance(tool_msg.additional_kwargs["duration"], int)

    async def test_multiple_tool_calls_all_success_async(self):
        """测试4: 模型返回多个工具调用，所有工具正常执行（异步）"""
        tool_node = build_tool_node(tools=[calculator, multiplier])

        # 构造状态：包含两个工具调用
        state = {
            "messages": [
                HumanMessage(content="Calculate 2+3 and 4*5"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"a": 2, "b": 3},
                            "id": "call_1",
                            "type": "tool_call",
                        },
                        {
                            "name": "multiplier",
                            "args": {"a": 4, "b": 5},
                            "id": "call_2",
                            "type": "tool_call",
                        },
                    ]
                ),
            ]
        }

        # 异步执行 tool_node
        result = await arun_tool_node_in_graph(tool_node, state)

        # 验证结果
        assert isinstance(result, dict)
        assert "messages" in result

        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 2

        # 验证第一个工具调用
        calc_msg = next(msg for msg in tool_messages if msg.name == "calculator")
        assert calc_msg.content == "5"
        assert calc_msg.tool_call_id == "call_1"
        assert "duration" in calc_msg.additional_kwargs
        assert "description" in calc_msg.additional_kwargs

        # 验证第二个工具调用
        mult_msg = next(msg for msg in tool_messages if msg.name == "multiplier")
        assert mult_msg.content == "20"
        assert mult_msg.tool_call_id == "call_2"
        assert "duration" in mult_msg.additional_kwargs
        assert "description" in mult_msg.additional_kwargs

    async def test_multiple_tool_calls_with_failures_async(self):
        """测试5: 模型返回多个工具调用，部分工具抛出异常（异步）"""
        tool_node = build_tool_node(
            tools=[calculator, failing_tool],
            handle_tool_errors=True
        )

        # 构造状态：包含一个成功和一个失败的工具调用
        state = {
            "messages": [
                HumanMessage(content="Test mixed results"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"a": 2, "b": 3},
                            "id": "call_success",
                            "type": "tool_call",
                        },
                        {
                            "name": "failing_tool",
                            "args": {"message": "test"},
                            "id": "call_fail",
                            "type": "tool_call",
                        },
                    ]
                ),
            ]
        }

        # 异步执行 tool_node
        result = await arun_tool_node_in_graph(tool_node, state)

        # 验证结果
        assert isinstance(result, dict)
        assert "messages" in result

        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 2

        # 验证成功的工具调用
        success_msg = next(msg for msg in tool_messages if msg.name == "calculator")
        assert success_msg.content == "5"
        assert success_msg.tool_call_id == "call_success"
        assert "duration" in success_msg.additional_kwargs

        # 验证失败的工具调用
        fail_msg = next(msg for msg in tool_messages if msg.name == "failing_tool")
        assert "Tool failed: test" in fail_msg.content
        assert fail_msg.tool_call_id == "call_fail"
        assert fail_msg.status == "error"
        assert "duration" in fail_msg.additional_kwargs

    async def test_timing_accuracy_async(self):
        """测试6: 验证异步执行时间记录的准确性"""
        tool_node = build_tool_node(tools=[slow_tool])

        # 构造状态：调用慢速工具
        state = {
            "messages": [
                HumanMessage(content="Test timing"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "slow_tool",
                            "args": {"duration_ms": 100},
                            "id": "call_slow",
                            "type": "tool_call",
                        }
                    ]
                ),
            ]
        }

        # 异步执行 tool_node
        result = await arun_tool_node_in_graph(tool_node, state)

        # 验证结果
        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 1

        tool_msg = tool_messages[0]
        assert "duration" in tool_msg.additional_kwargs

        # 执行时间应该大约是 100ms (允许一些误差)
        duration = tool_msg.additional_kwargs["duration"]
        assert duration >= 90  # 至少 90ms
        assert duration <= 200  # 最多 200ms (留一些余量)

    async def test_handle_tool_errors_true_async(self):
        """测试7: 当 handle_tool_errors=True 时，异步异常不应该被抛出"""
        tool_node = build_tool_node(
            tools=[failing_tool],
            handle_tool_errors=True
        )

        # 构造状态：包含一个会失败的工具调用
        state = {
            "messages": [
                HumanMessage(content="Test error handling"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "failing_tool",
                            "args": {"message": "test"},
                            "id": "call_fail",
                            "type": "tool_call",
                        }
                    ]
                ),
            ]
        }

        # 异步执行 tool_node，不应该抛出异常
        result = await arun_tool_node_in_graph(tool_node, state)

        # 验证结果包含错误信息但没有抛出异常
        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].status == "error"
        assert "Tool failed: test" in tool_messages[0].content

    async def test_custom_async_wrapper(self):
        """测试8: 自定义异步包装器是否正常工作"""
        call_log = []

        async def async_logging_wrapper(request, execute):
            call_log.append(f"async_before:{request.tool_call['name']}")
            result = await execute(request)
            call_log.append(f"async_after:{request.tool_call['name']}")
            return result

        tool_node = build_tool_node(
            tools=[calculator],
            async_wrappers=[async_logging_wrapper]
        )

        # 构造状态
        state = {
            "messages": [
                HumanMessage(content="Test async wrapper"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"a": 2, "b": 3},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ]
                ),
            ]
        }

        # 异步执行 tool_node
        result = await arun_tool_node_in_graph(tool_node, state)

        # 验证自定义包装器被调用
        assert len(call_log) == 2
        assert call_log[0] == "async_before:calculator"
        assert call_log[1] == "async_after:calculator"

        # 验证结果仍然正确
        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].content == "5"

        # 验证元数据仍然存在（timer_wrapper 在自定义 wrapper 之前执行）
        assert "duration" in tool_messages[0].additional_kwargs
        assert "description" in tool_messages[0].additional_kwargs

    async def test_multiple_custom_async_wrappers(self):
        """测试8+: 多个自定义异步包装器的执行顺序"""
        execution_order = []

        async def async_wrapper1(request, execute):
            execution_order.append("async_wrapper1_start")
            result = await execute(request)
            execution_order.append("async_wrapper1_end")
            return result

        async def async_wrapper2(request, execute):
            execution_order.append("async_wrapper2_start")
            result = await execute(request)
            execution_order.append("async_wrapper2_end")
            return result

        tool_node = build_tool_node(
            tools=[calculator],
            async_wrappers=[async_wrapper1, async_wrapper2]
        )

        # 构造状态
        state = {
            "messages": [
                HumanMessage(content="Test async wrapper order"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"a": 1, "b": 1},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ]
                ),
            ]
        }

        # 异步执行 tool_node
        result = await arun_tool_node_in_graph(tool_node, state)

        # 验证执行顺序：timer -> wrapper1 -> wrapper2 -> tool -> wrapper2 -> wrapper1 -> timer
        # (我们只能看到 wrapper1 和 wrapper2 的顺序)
        assert execution_order == [
            "async_wrapper1_start",
            "async_wrapper2_start",
            "async_wrapper2_end",
            "async_wrapper1_end",
        ]

    async def test_async_wrapper_can_modify_request(self):
        """测试8++: 异步包装器可以修改请求"""
        async def async_doubling_wrapper(request, execute):
            # 将参数翻倍
            modified_request = request.override(
                tool_call={
                    **request.tool_call,
                    "args": {
                        "a": request.tool_call["args"]["a"] * 2,
                        "b": request.tool_call["args"]["b"] * 2,
                    }
                }
            )
            return await execute(modified_request)

        tool_node = build_tool_node(
            tools=[calculator],
            async_wrappers=[async_doubling_wrapper]
        )

        # 构造状态：原始请求是 2 + 3
        state = {
            "messages": [
                HumanMessage(content="Test request modification"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"a": 2, "b": 3},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ]
                ),
            ]
        }

        # 异步执行 tool_node
        result = await arun_tool_node_in_graph(tool_node, state)

        # 验证结果：应该是 (2*2) + (3*2) = 4 + 6 = 10
        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].content == "10"

    async def test_mixed_sync_async_wrappers_async(self):
        """测试9: 同时有同步和异步包装器时，异步执行应该只使用异步包装器"""
        sync_call_log = []
        async_call_log = []

        def sync_wrapper(request, execute):
            sync_call_log.append(f"sync:{request.tool_call['name']}")
            return execute(request)

        async def async_wrapper(request, execute):
            async_call_log.append(f"async:{request.tool_call['name']}")
            result = await execute(request)
            return result

        tool_node = build_tool_node(
            tools=[calculator],
            wrappers=[sync_wrapper],
            async_wrappers=[async_wrapper]
        )

        # 构造状态
        state = {
            "messages": [
                HumanMessage(content="Test mixed wrappers"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"a": 2, "b": 3},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ]
                ),
            ]
        }

        # 异步执行 tool_node
        result = await arun_tool_node_in_graph(tool_node, state)

        # 验证：异步执行时，只有异步包装器被调用
        assert len(async_call_log) == 1
        assert async_call_log[0] == "async:calculator"
        assert len(sync_call_log) == 0  # 同步包装器不应该被调用

        # 验证结果仍然正确
        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].content == "5"
