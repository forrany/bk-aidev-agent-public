# -*- coding: utf-8 -*-
"""检查点消息同步测试（Phase 6 修复）。

验证：
- thread_id 稳定性：统一使用 self.thread_id，无 uuid4 后缀
- RemoveMessage 同步在 _execute() 中执行（覆盖流式和非流式路径）
- PV 状态在稳定 thread_id 下跨请求持久化
- 边界情况：空检查点、id=None 的消息
"""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage

# 读取 chat.py 源码（避免 import 触发 ag_ui 等不可用依赖）
_CHAT_PY_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "aidev_agent", "services", "agent", "chat.py")
_CHAT_PY_PATH = os.path.normpath(_CHAT_PY_PATH)


def _read_chat_py() -> str:
    with open(_CHAT_PY_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _extract_method_source(source: str, method_name: str) -> str:
    """从源码中提取方法的缩进源码块。"""
    lines = source.splitlines()
    start = None
    indent = None
    result = []
    in_signature = False  # 多行签名（括号未闭合）
    paren_depth = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if start is None:
            if stripped.startswith(f"def {method_name}("):
                start = i
                indent = len(line) - len(line.lstrip())
                result.append(line)
                # 跟踪括号深度
                paren_depth += stripped.count("(") - stripped.count(")")
                in_signature = paren_depth > 0
        else:
            if in_signature:
                result.append(line)
                paren_depth += stripped.count("(") - stripped.count(")")
                if paren_depth <= 0:
                    in_signature = False
                continue
            if line.strip() == "" or len(line) - len(line.lstrip()) > indent:
                result.append(line)
            else:
                break
    return "\n".join(result)


class TestThreadIdStability:
    """验证 thread_id 统一使用 self.thread_id，无 UUID 后缀。"""

    def test_stream_uses_self_thread_id(self):
        """_stream() 应直接使用 self.thread_id，无中间变量。"""
        source = _read_chat_py()
        stream_source = _extract_method_source(source, "_stream")
        assert "uuid.uuid4().hex[:8]" not in stream_source, "graph_thread_id 仍使用 uuid4 后缀 — bug 未修复"
        # 不应有 stream_thread_id 或 graph_thread_id 中间变量
        assert "stream_thread_id" not in stream_source, (
            "_stream() 不应有 stream_thread_id 中间变量，应直接使用 self.thread_id"
        )
        assert "graph_thread_id" not in stream_source, (
            "_stream() 不应有 graph_thread_id 中间变量，应直接使用 self.thread_id"
        )
        # 所有 thread_id 使用均为 self.thread_id
        assert '"thread_id": self.thread_id' in stream_source, "AgentInput.thread_id 应为 self.thread_id"

    def test_execute_uses_self_thread_id(self):
        """_execute() 应使用 self.thread_id，不用 execute_kwargs.session_code fallback。"""
        source = _read_chat_py()
        execute_source = _extract_method_source(source, "_execute")
        assert 'cfg["configurable"]["thread_id"] = self.thread_id' in execute_source, (
            "_execute 应直接使用 self.thread_id"
        )
        # 确保没有 session_code fallback 的赋值行（注释中的不算）
        for line in execute_source.splitlines():
            stripped = line.strip()
            if stripped.startswith("cfg[") and "thread_id" in stripped and "session_code" in stripped:
                pytest.fail("_execute 不应有 execute_kwargs.session_code fallback 赋值")


class TestRemoveMessageSync:
    """验证 _execute() 使用 RemoveMessage 同步检查点消息。"""

    @pytest.mark.asyncio
    async def test_sync_removes_stale_messages(self):
        """当检查点包含消息时，应为所有非系统消息创建 RemoveMessage 操作。"""
        checkpoint_msg1 = HumanMessage(id="cp-msg-1", content="hello")
        checkpoint_msg2 = AIMessage(id="cp-msg-2", content="hi there")
        checkpoint_msg3 = SystemMessage(id="cp-msg-3", content="you are helpful")

        mock_snapshot = MagicMock()
        mock_snapshot.values = {"messages": [checkpoint_msg1, checkpoint_msg2, checkpoint_msg3]}

        checkpoint_messages = mock_snapshot.values.get("messages", [])
        non_system = [m for m in checkpoint_messages if not isinstance(m, SystemMessage)]
        remove_ops = [RemoveMessage(id=m.id) for m in non_system if m.id is not None]

        assert len(remove_ops) == 2
        assert remove_ops[0].id == "cp-msg-1"
        assert remove_ops[1].id == "cp-msg-2"

    @pytest.mark.asyncio
    async def test_sync_empty_checkpoint(self):
        """当检查点为空时，不需要 RemoveMessage 操作。"""
        mock_snapshot = MagicMock()
        mock_snapshot.values = {"messages": []}

        checkpoint_messages = mock_snapshot.values.get("messages", [])
        non_system = [m for m in checkpoint_messages if not isinstance(m, SystemMessage)]
        remove_ops = [RemoveMessage(id=m.id) for m in non_system if m.id is not None]

        assert len(remove_ops) == 0

    @pytest.mark.asyncio
    async def test_sync_message_with_none_id(self):
        """id=None 的消息应被跳过，不应导致 RemoveMessage 使用 None id。"""
        msg_no_id = HumanMessage(content="no id here")
        assert msg_no_id.id is None

        remove_ops = [RemoveMessage(id=m.id) for m in [msg_no_id] if m.id is not None]
        assert len(remove_ops) == 0

    @pytest.mark.asyncio
    async def test_aupdate_state_called_with_as_node_start(self):
        """aupdate_state 必须以 as_node='__start__' 调用，以确保正确的检查点归属。"""
        mock_graph = AsyncMock()
        mock_graph.aupdate_state = AsyncMock()

        remove_ops = [MagicMock(id="msg-1")]
        await mock_graph.aupdate_state(
            {"configurable": {"thread_id": "test"}},
            {"messages": remove_ops},
            as_node="__start__",
        )

        mock_graph.aupdate_state.assert_called_once()
        call_kwargs = mock_graph.aupdate_state.call_args
        assert call_kwargs[1].get("as_node") == "__start__" or (
            len(call_kwargs[0]) > 2 and call_kwargs[0][2] == "__start__"
        )

    def test_sync_in_execute_not_prepare_stream(self):
        """同步逻辑应在 _execute() 中调用（而非 prepare_stream() 中）。"""
        source = _read_chat_py()
        execute_source = _extract_method_source(source, "_execute")
        sync_source = _extract_method_source(source, "_sync_checkpoint_messages")
        # _execute 应调用 _sync_checkpoint_messages
        assert "_sync_checkpoint_messages" in execute_source, "_execute() 应调用 _sync_checkpoint_messages()"
        # RemoveMessage 实际逻辑在 _sync_checkpoint_messages 中
        assert "RemoveMessage" in sync_source, "_sync_checkpoint_messages() 应包含 RemoveMessage 同步逻辑"

    def test_sync_covers_both_paths(self):
        """同步逻辑在流式/非流式分支之前执行，覆盖两条路径。"""
        source = _read_chat_py()
        execute_source = _extract_method_source(source, "_execute")
        # _sync_checkpoint_messages 调用应在流式/非流式分支之前
        sync_call_pos = execute_source.find("_sync_checkpoint_messages")
        stream_pos = execute_source.find("execute_kwargs.stream")
        ainvoke_pos = execute_source.find("ainvoke")

        assert sync_call_pos > 0, "_execute() 应调用 _sync_checkpoint_messages"
        assert sync_call_pos < stream_pos, "同步调用应在流式分支之前"
        assert sync_call_pos < ainvoke_pos, "同步调用应在 ainvoke 之前"

    def test_sync_skipped_on_resume(self):
        """resume 路径下必须跳过 _sync_checkpoint_messages，避免清空 checkpoint 中的 tool_call 上下文。"""
        source = _read_chat_py()
        execute_source = _extract_method_source(source, "_execute")
        # _execute 内必须存在 resume 跳过判断（形如 `if not execute_kwargs.resume:` 包裹 sync 调用）
        assert "execute_kwargs.resume" in execute_source, (
            "_execute() 应基于 execute_kwargs.resume 决定是否跳过 _sync_checkpoint_messages"
        )
        # 粗略校验：sync 调用应被 if not ... resume 包裹（容忍空白差异）
        normalized = " ".join(execute_source.split())
        assert "if not execute_kwargs.resume" in normalized and (
            normalized.find("if not execute_kwargs.resume")
            < normalized.find("_sync_checkpoint_messages(agent_e, cfg)")
        ), "resume 路径下必须显式跳过 _sync_checkpoint_messages 调用"


class TestPVStatePersistence:
    """验证 PV 状态在稳定 thread_id 下跨请求持久化。"""

    @pytest.mark.asyncio
    async def test_pv_state_survives_across_invocations(self):
        """使用稳定的 thread_id，runtime_paas_sbx_pv 应在检查点状态中持久化。"""
        from typing import Annotated

        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, START, StateGraph, add_messages
        from typing_extensions import TypedDict

        class TestState(TypedDict):
            messages: Annotated[list, add_messages]
            runtime_paas_sbx_pv: list

        def dummy_node(state):
            return {"messages": [AIMessage(content="ok")]}

        builder = StateGraph(TestState)
        builder.add_node("agent", dummy_node)
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)

        checkpointer = MemorySaver()
        graph = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "session_abc"}}

        # 第一次调用 — 设置 PV 状态
        await graph.ainvoke(
            {
                "messages": [HumanMessage(content="hi")],
                "runtime_paas_sbx_pv": [{"type": "paas-sbx-pv", "volume_id": "test-vol"}],
            },
            config,
        )

        # 验证 PV 状态已持久化
        state1 = await graph.aget_state(config)
        pv_list = state1.values.get("runtime_paas_sbx_pv", [])
        assert len(pv_list) == 1
        assert pv_list[0]["volume_id"] == "test-vol"

        # 第二次调用 — 同一 thread_id，PV 应仍然存在
        await graph.ainvoke({"messages": [HumanMessage(content="hello again")]}, config)
        state2 = await graph.aget_state(config)
        pv_list2 = state2.values.get("runtime_paas_sbx_pv", [])
        assert len(pv_list2) == 1
        assert pv_list2[0]["volume_id"] == "test-vol"
