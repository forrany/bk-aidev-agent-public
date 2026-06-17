# -*- coding: utf-8 -*-
"""LocalBackend 单元测试。

覆盖：
1. LocalBackend 类零参构造
2. 满足 AgentBackend Protocol
3. 缺少 agent_cls 时 raise ValueError
4. 缺少 ctx 时 raise ValueError
5. execute() 正常执行 agent_cls() + build(ctx) + child.execute()
6. execute() 注入 session_code 和 session_context_data 到 ctx
7. 嵌套保护标志 _A2A_SUBAGENT_FLAG 传播
8. child.execute() 异常时返回 failed 字典，不抛异常
9. agent_cls().build(ctx) 异常时返回 failed 字典，不抛异常
10. _extract_text() 从 choices 中提取文本
11. _extract_text() 兼容 str 返回
12. _extract_text() 无法识别格式返回空字符串
13. _run_subagent 流式执行
14. _run_subagent 超时诊断
15. _run_subagent 心跳发送
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from aidev_agent.core.tools.a2a_tools.local_backend import (
    _ERR_MISSING_AGENT_INSTANCE,
    _ERR_MISSING_CTX,
    LocalBackend,
)
from aidev_agent.core.tools.a2a_tools.progress import count_tool_calls
from aidev_agent.core.tools.a2a_tools.provider import _A2A_SUBAGENT_FLAG
from aidev_agent.core.tools.a2a_tools.types import AgentBackend, AgentBackendType, AgentSpec


def _make_spec(**overrides: Any) -> AgentSpec:
    """创建测试用 AgentSpec，默认 backend_type=LOCAL。"""
    defaults: dict[str, Any] = {
        "name": "test_local_agent",
        "description": "A test local agent",
        "backend_type": AgentBackendType.LOCAL,
        "params": {},
        "timeout_seconds": 300,
    }
    defaults.update(overrides)
    return AgentSpec(**defaults)


def _make_mock_rm() -> MagicMock:
    """创建 mock resource_manager。"""
    return MagicMock()


@dataclass
class _FakeCtx:
    """模拟 AgentBuildContext（dataclass，支持 dataclasses.replace）。"""

    session_code: str = ""
    session_context_data: list[dict] = field(default_factory=list)
    resource_manager: Any = field(default_factory=_make_mock_rm)
    extra: dict[str, Any] = field(default_factory=dict)


def _make_mock_child(result_text: str = "sub-agent result") -> MagicMock:
    """创建 mock child agent，execute 返回标准格式。"""
    child = MagicMock()
    child.execute.return_value = {
        "choices": [{"delta": {"role": "assistant", "content": result_text}}],
    }
    child.thread_id = "test_thread_id"
    return child


def _make_mock_agent_cls(mock_child: MagicMock | None = None) -> MagicMock:
    """创建 mock agent_cls，调用时返回实例（其 build 返回 mock_child）。"""
    if mock_child is None:
        mock_child = _make_mock_child()

    mock_instance = MagicMock()
    mock_instance.build.return_value = mock_child

    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls


# ============== Test 1: 零参构造 ==============


class TestLocalBackendConstruction:
    """Test 1: LocalBackend 类零参构造成功。"""

    def test_zero_arg_construction(self) -> None:
        backend = LocalBackend()
        assert backend is not None


# ============== Test 2: 满足 AgentBackend Protocol ==============


class TestLocalBackendProtocol:
    """Test 2: LocalBackend 满足 AgentBackend Protocol (isinstance 检查)。"""

    def test_satisfies_agent_backend_protocol(self) -> None:
        backend = LocalBackend()
        assert isinstance(backend, AgentBackend)


# ============== Test 3: 缺少 agent_cls ==============


class TestLocalBackendMissingAgentCls:
    """Test 3: execute() 缺少 agent_cls 时 raise ValueError。"""

    def test_missing_agent_cls_raises_error(self) -> None:
        backend = LocalBackend()
        spec = _make_spec(params={"ctx": _FakeCtx()})
        with pytest.raises(ValueError, match=_ERR_MISSING_AGENT_INSTANCE):
            backend.execute(spec, "hello", session_code="sess-test")

    def test_none_agent_cls_raises_error(self) -> None:
        backend = LocalBackend()
        spec = _make_spec(params={"agent_cls": None, "ctx": _FakeCtx()})
        with pytest.raises(ValueError, match=_ERR_MISSING_AGENT_INSTANCE):
            backend.execute(spec, "hello", session_code="sess-test")


# ============== Test 4: 缺少 ctx ==============


class TestLocalBackendMissingCtx:
    """Test 4: execute() 缺少 ctx 时 raise ValueError。"""

    def test_missing_ctx_raises_error(self) -> None:
        backend = LocalBackend()
        mock_cls = _make_mock_agent_cls()
        spec = _make_spec(params={"agent_cls": mock_cls})
        with pytest.raises(ValueError, match=_ERR_MISSING_CTX):
            backend.execute(spec, "hello", session_code="sess-test")

    def test_none_ctx_raises_error(self) -> None:
        backend = LocalBackend()
        mock_cls = _make_mock_agent_cls()
        spec = _make_spec(params={"agent_cls": mock_cls, "ctx": None})
        with pytest.raises(ValueError, match=_ERR_MISSING_CTX):
            backend.execute(spec, "hello", session_code="sess-test")


# ============== Test 5: execute() 正常执行 ==============


class TestLocalBackendNormalExecution:
    """Test 5: execute() 正常调用 agent_cls() + build(ctx) + child.execute()，返回结果。"""

    def test_normal_execution_returns_completed(self) -> None:
        backend = LocalBackend()

        mock_child = _make_mock_child("sub-agent result")
        mock_cls = _make_mock_agent_cls(mock_child)

        ctx = _FakeCtx()
        spec = _make_spec(params={"agent_cls": mock_cls, "ctx": ctx})
        result = backend.execute(spec, "do something", session_code="sess-test")

        assert result.status == "completed"
        assert result.result == "sub-agent result"
        # 验证 agent_cls 被调用创建实例
        mock_cls.assert_called_once()
        # 验证 build 被调用
        mock_cls.return_value.build.assert_called_once()
        mock_child.execute.assert_called_once()

    def test_execute_injects_session_context_data(self) -> None:
        """验证 prepare_session 加载的 session_context_data 通过 ctx 传入 build。"""
        backend = LocalBackend()

        mock_child = _make_mock_child("result")
        mock_cls = _make_mock_agent_cls(mock_child)

        # 模拟 resource_manager 返回历史上下文（含当前用户消息）
        mock_rm = MagicMock()
        mock_rm.get_chat_session_context.return_value = [
            {"role": "user", "content": "test message"},
        ]

        ctx = _FakeCtx(resource_manager=mock_rm)
        spec = _make_spec(params={"agent_cls": mock_cls, "ctx": ctx})
        backend.execute(spec, "test message", session_code="sess-test")

        # 验证 build 传入的 ctx 包含 session_context_data
        call_args = mock_cls.return_value.build.call_args
        built_ctx = call_args[0][0]
        assert built_ctx.session_context_data == [{"role": "user", "content": "test message"}]

    def test_execute_missing_rm_raises_error(self) -> None:
        """ctx 没有 resource_manager 时 raise ValueError。"""
        backend = LocalBackend()
        mock_cls = _make_mock_agent_cls()
        ctx = _FakeCtx(resource_manager=None)
        spec = _make_spec(params={"agent_cls": mock_cls, "ctx": ctx})
        with pytest.raises(ValueError, match="resource_manager"):
            backend.execute(spec, "hello", session_code="sess-test")


# ============== Test 6: execute() session_code 注入 ==============


class TestLocalBackendSessionCodeRequired:
    """Test 6: execute() 要求 session_code 非空，并正确注入到 ctx。"""

    def test_empty_session_code_raises_error(self) -> None:
        """空 session_code 触发 ValueError。"""
        backend = LocalBackend()

        mock_cls = _make_mock_agent_cls()

        ctx = _FakeCtx()
        spec = _make_spec(params={"agent_cls": mock_cls, "ctx": ctx})

        with pytest.raises(ValueError, match="session_code must not be empty"):
            backend.execute(spec, "hello", session_code="")

    def test_execute_member_returns_completed_with_session_code(self) -> None:
        backend = LocalBackend()

        mock_cls = _make_mock_agent_cls()

        ctx = _FakeCtx()
        spec = _make_spec(params={"agent_cls": mock_cls, "ctx": ctx})
        result = backend.execute(spec, "hello", session_code="sess-123")

        assert result.status == "completed"
        assert result.result == "sub-agent result"

    def test_execute_member_injects_session_code_into_ctx(self) -> None:
        """验证 session_code 通过 dataclasses.replace 注入到 ctx。"""
        backend = LocalBackend()

        mock_cls = _make_mock_agent_cls()

        ctx = _FakeCtx()
        spec = _make_spec(params={"agent_cls": mock_cls, "ctx": ctx})
        backend.execute(spec, "hello", session_code="sess-456")

        # build 传入的 ctx 的 session_code 应被替换
        call_args = mock_cls.return_value.build.call_args
        built_ctx = call_args[0][0]
        assert built_ctx.session_code == "sess-456"

    def test_execute_member_missing_agent_cls_raises_error(self) -> None:
        """缺少 agent_cls 时 raise ValueError（validate 在 session_code 之后）。"""
        backend = LocalBackend()
        spec = _make_spec(params={"ctx": _FakeCtx()})
        with pytest.raises(ValueError, match=_ERR_MISSING_AGENT_INSTANCE):
            backend.execute(spec, "hello", session_code="sess-789")


# ============== Test 7: 嵌套保护标志传播 ==============


class TestLocalBackendNestingProtection:
    """Test 7: execute() 传播嵌套保护标志。"""

    def test_execute_propagates_subagent_flag(self) -> None:
        backend = LocalBackend()

        mock_cls = _make_mock_agent_cls()

        ctx = _FakeCtx()
        spec = _make_spec(params={"agent_cls": mock_cls, "ctx": ctx})
        backend.execute(spec, "hello", session_code="sess-flag")

        # 验证 build 传入的 ctx.extra 包含嵌套保护标志
        call_args = mock_cls.return_value.build.call_args
        built_ctx = call_args[0][0]
        assert built_ctx.extra.get(_A2A_SUBAGENT_FLAG) is True

    def test_execute_preserves_existing_extra(self) -> None:
        """已有 extra 内容不被覆盖。"""
        backend = LocalBackend()

        mock_cls = _make_mock_agent_cls()

        ctx = _FakeCtx(extra={"existing_key": "existing_value"})
        spec = _make_spec(params={"agent_cls": mock_cls, "ctx": ctx})
        backend.execute(spec, "hello", session_code="sess-extra")

        call_args = mock_cls.return_value.build.call_args
        built_ctx = call_args[0][0]
        assert built_ctx.extra.get("existing_key") == "existing_value"
        assert built_ctx.extra.get(_A2A_SUBAGENT_FLAG) is True

    def test_execute_member_propagates_both_session_and_flag(self) -> None:
        """execute 同时注入 session_code 和嵌套标志。"""
        backend = LocalBackend()

        mock_cls = _make_mock_agent_cls()

        ctx = _FakeCtx()
        spec = _make_spec(params={"agent_cls": mock_cls, "ctx": ctx})
        backend.execute(spec, "hello", session_code="sess-abc")

        call_args = mock_cls.return_value.build.call_args
        built_ctx = call_args[0][0]
        assert built_ctx.session_code == "sess-abc"
        assert built_ctx.extra.get(_A2A_SUBAGENT_FLAG) is True


# ============== Test 8: child.execute() 异常 ==============


class TestLocalBackendExecutionError:
    """Test 8: execute() 执行异常时返回 failed 字典，不抛异常。"""

    def test_execute_exception_returns_failed(self) -> None:
        backend = LocalBackend()

        mock_child = MagicMock()
        mock_child.execute.side_effect = RuntimeError("child execution failed")
        mock_child.thread_id = "test_thread_id"

        mock_cls = _make_mock_agent_cls(mock_child)

        ctx = _FakeCtx()
        spec = _make_spec(params={"agent_cls": mock_cls, "ctx": ctx}, timeout_seconds=5)
        result = backend.execute(spec, "hello", session_code="sess-err")

        assert result.status == "failed"
        assert "child execution failed" in result.error


# ============== Test 9: agent_cls().build() 异常 ==============


class TestLocalBackendBuildError:
    """Test 9: agent_cls().build(ctx) 异常时返回 failed 字典，不抛异常。"""

    def test_build_exception_returns_failed(self) -> None:
        backend = LocalBackend()

        mock_instance = MagicMock()
        mock_instance.build.side_effect = ValueError("build config error")

        mock_cls = MagicMock(return_value=mock_instance)

        ctx = _FakeCtx()
        spec = _make_spec(params={"agent_cls": mock_cls, "ctx": ctx})
        result = backend.execute(spec, "hello", session_code="sess-build-err")

        assert result.status == "failed"
        assert "build config error" in result.error


# ============== Test 10: _extract_text 从 choices 提取文本 ==============


class TestExtractText:
    """Test 10: _extract_text() 从标准 choices 格式中提取文本。"""

    def test_extract_from_choices(self) -> None:
        result = {"choices": [{"delta": {"role": "assistant", "content": "hello world"}}]}
        assert LocalBackend._extract_text(result) == "hello world"

    def test_extract_empty_choices(self) -> None:
        result: dict[str, Any] = {"choices": []}
        assert LocalBackend._extract_text(result) == ""

    def test_extract_missing_delta(self) -> None:
        result = {"choices": [{}]}
        assert LocalBackend._extract_text(result) == ""

    def test_extract_none_content(self) -> None:
        result = {"choices": [{"delta": {"role": "assistant", "content": None}}]}
        assert LocalBackend._extract_text(result) == ""


# ============== Test 11: _extract_text 兼容 str 返回 ==============


class TestExtractTextStrFallback:
    """Test 11: _extract_text() 兼容 str 返回。"""

    def test_str_result(self) -> None:
        assert LocalBackend._extract_text("plain text result") == "plain text result"


# ============== Test 12: _extract_text 无法识别格式 ==============


class TestExtractTextUnknownFormat:
    """Test 12: _extract_text() 无法识别格式返回空字符串。"""

    def test_unknown_format_returns_empty(self) -> None:
        assert LocalBackend._extract_text(12345) == ""

    def test_none_result_returns_empty(self) -> None:
        assert LocalBackend._extract_text(None) == ""

    def test_empty_dict_returns_empty(self) -> None:
        assert LocalBackend._extract_text({}) == ""


# ============== Test 13: _run_subagent 流式执行 ==============


class TestRunSubagent:
    """Test 13: _run_subagent 流式执行子 Agent。"""

    def test_run_subagent_calls_execute_with_stream(self) -> None:
        """验证 _run_subagent 使用 stream 模式调用 subagent.execute。"""
        backend = LocalBackend()

        mock_subagent = MagicMock()
        mock_subagent.execute.return_value = []  # 流式返回空事件列表

        result_text, events = backend._run_subagent(mock_subagent, timeout_seconds=60)  # noqa: SLF001

        # 验证 subagent.execute 被调用（execute_kwargs 由 _run_subagent 内部构造）
        mock_subagent.execute.assert_called_once()

        assert result_text == ""
        assert events == []

    def test_run_subagent_with_existing_execute_kwargs(self) -> None:
        """验证 _run_subagent 使用传入的 execute_kwargs（仅设置 invoke_timeout 若为 None）。"""
        from aidev_agent.pydantic_models import ExecuteKwargs

        backend = LocalBackend()

        mock_subagent = MagicMock()
        mock_subagent.execute.return_value = []  # 流式返回空事件列表

        ek = ExecuteKwargs(stream=True, invoke_timeout=None)
        result_text, events = backend._run_subagent(  # noqa: SLF001
            mock_subagent, timeout_seconds=60, execute_kwargs=ek
        )

        # _run_subagent 在 invoke_timeout 为 None 时设置 timeout_seconds
        assert ek.invoke_timeout == 60
        # stream 由 execute/_extract_execute_kwargs 处理，_run_subagent 不再修改
        assert ek.stream is True
        mock_subagent.execute.assert_called_once_with(ek)
        assert result_text == ""


# ============== Test 14: _run_subagent 超时诊断 ==============


class TestLocalBackendTimeoutWarning:
    """Test 14: _run_subagent 超时时输出结构化 logger.warning（D-01）。"""

    def test_timeout_emits_warning_with_correct_fields(self) -> None:
        """当流式迭代中抛出 TimeoutError 时，调用 logger.warning 输出结构化信息。"""
        backend = LocalBackend()

        mock_subagent = MagicMock()
        mock_subagent.name = "test_timeout_agent"

        # subagent.execute() 返回一个迭代器，迭代时抛出 TimeoutError（模拟流式超时）
        mock_result = MagicMock()
        mock_result.__iter__.side_effect = TimeoutError("execution timed out")
        mock_subagent.execute.return_value = mock_result

        with (
            patch("aidev_agent.core.tools.a2a_tools.local_backend.logger") as mock_logger,
            pytest.raises(TimeoutError),
        ):
            backend._run_subagent(mock_subagent, timeout_seconds=5)  # noqa: SLF001

        # 验证 logger.warning 被调用
        assert mock_logger.warning.called

        # 获取 warning 调用参数
        call_args = mock_logger.warning.call_args
        format_string = call_args[0][0]
        args = call_args[0][1:]

        # 验证结构化字段
        assert "A2A subagent timeout" in format_string
        assert "agent_name=" in format_string
        assert "timeout_seconds=" in format_string
        assert "backend_type=local" in format_string

        # 验证 agent_name 通过参数传入
        assert "test_timeout_agent" in str(args)

    def test_timeout_error_re_raised(self) -> None:
        """TimeoutError 在日志记录后重新抛出。"""
        backend = LocalBackend()

        mock_subagent = MagicMock()

        # subagent.execute() 返回一个迭代器，迭代时抛出 TimeoutError
        mock_result = MagicMock()
        mock_result.__iter__.side_effect = TimeoutError("timed out")
        mock_subagent.execute.return_value = mock_result

        with (
            patch("aidev_agent.core.tools.a2a_tools.local_backend.logger"),
            pytest.raises(TimeoutError, match="timed out"),
        ):
            backend._run_subagent(mock_subagent, timeout_seconds=10)  # noqa: SLF001

    def test_normal_execution_no_timeout_warning(self) -> None:
        """正常执行路径不输出超时警告。"""
        backend = LocalBackend()

        mock_subagent = MagicMock()
        mock_subagent.execute.return_value = [
            'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "Hello"}',
            "data: [DONE]",
        ]

        with patch("aidev_agent.core.tools.a2a_tools.local_backend.logger") as mock_logger:
            result_text, events = backend._run_subagent(mock_subagent, timeout_seconds=30)  # noqa: SLF001

            # 检查 warning 调用中是否有超时相关消息
            timeout_warning_found = any(
                "A2A subagent timeout" in str(call) for call in mock_logger.warning.call_args_list
            )
            assert not timeout_warning_found
            assert isinstance(result_text, str)

    def test_non_timeout_error_no_timeout_warning(self) -> None:
        """非 TimeoutError 异常不输出超时警告。"""
        backend = LocalBackend()

        mock_subagent = MagicMock()

        # subagent.execute() 返回一个迭代器，迭代时抛出 RuntimeError
        mock_result = MagicMock()
        mock_result.__iter__.side_effect = RuntimeError("something else broke")
        mock_subagent.execute.return_value = mock_result

        with (
            patch("aidev_agent.core.tools.a2a_tools.local_backend.logger") as mock_logger,
            pytest.raises(RuntimeError),
        ):
            backend._run_subagent(mock_subagent, timeout_seconds=5)  # noqa: SLF001

        # 不应有 A2A subagent timeout 的 warning
        timeout_warning_found = any("A2A subagent timeout" in str(call) for call in mock_logger.warning.call_args_list)
        assert not timeout_warning_found


# ============== Test 15: _run_subagent 心跳发送 ==============


class TestLocalBackendHeartbeat:
    """Test 15: _run_subagent 流式循环中发送 subagent.heartbeat 事件（D-04）。"""

    def test_heartbeat_events_sent_with_callback(self) -> None:
        """当 progress_callback 为 mock 时，流式执行中发送 subagent.heartbeat 事件。"""
        backend = LocalBackend()

        mock_subagent = MagicMock()
        mock_subagent.execute.return_value = [
            'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "Hello"}',
            'data: {"type": "TOOL_CALL_START", "tool_call_id": "t1"}',
            "data: [DONE]",
        ]

        mock_callback = MagicMock()
        with patch("aidev_agent.core.tools.a2a_tools.local_backend.logger"):
            backend._run_subagent(  # noqa: SLF001
                mock_subagent, timeout_seconds=30, progress_callback=mock_callback
            )

        # 验证 progress_callback 被调用，且至少有一次带有 subagent.heartbeat
        heartbeat_calls = [call for call in mock_callback.call_args_list if call[0][0] == "subagent.heartbeat"]
        assert len(heartbeat_calls) > 0, "Expected at least one subagent.heartbeat event"

        # 验证第一个心跳包含正确字段（不再有 elapsed_seconds）
        first_heartbeat = heartbeat_calls[0]
        kwargs = first_heartbeat[1]
        assert "tool_count" in kwargs
        assert "iteration" in kwargs
        assert isinstance(kwargs["tool_count"], int)
        assert isinstance(kwargs["iteration"], int)

    def test_heartbeat_fields_correct(self) -> None:
        """每个心跳包含正确的 tool_count, iteration 值。"""
        backend = LocalBackend()

        mock_subagent = MagicMock()
        mock_subagent.execute.return_value = [
            'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "A"}',
            'data: {"type": "TOOL_CALL_START", "tool_call_id": "t1"}',
            'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "B"}',
            "data: [DONE]",
        ]

        mock_callback = MagicMock()
        with patch("aidev_agent.core.tools.a2a_tools.local_backend.logger"):
            backend._run_subagent(  # noqa: SLF001
                mock_subagent, timeout_seconds=30, progress_callback=mock_callback
            )

        heartbeat_calls = [call for call in mock_callback.call_args_list if call[0][0] == "subagent.heartbeat"]
        # 3 parsed events → 3 heartbeat calls
        assert len(heartbeat_calls) == 3

        # 最后一个心跳的 iteration 应该等于总事件数
        last_kwargs = heartbeat_calls[-1][1]
        assert last_kwargs["iteration"] == 3
        # tool_count 应该是 TOOL_CALL_START 事件总数
        assert last_kwargs["tool_count"] == 1

    def test_no_heartbeat_when_callback_none(self) -> None:
        """当 progress_callback 为 None 时，无心跳（向后兼容）。"""
        backend = LocalBackend()

        mock_subagent = MagicMock()
        mock_subagent.execute.return_value = [
            'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "Hello"}',
            "data: [DONE]",
        ]

        with patch("aidev_agent.core.tools.a2a_tools.local_backend.logger"):
            result_text, events = backend._run_subagent(mock_subagent, timeout_seconds=30)  # noqa: SLF001

        assert isinstance(result_text, str)
        assert len(events) == 1
        # 没有传入 callback → 无心跳，不报错

    def test_heartbeat_once_per_sse_event(self) -> None:
        """每个解析后的 SSE 事件触发一次心跳。"""
        backend = LocalBackend()

        # 返回 4 个有效 SSE 事件 + 空行和 [DONE]
        mock_subagent = MagicMock()
        mock_subagent.execute.return_value = [
            'data: {"type": "RUN_STARTED"}',
            "",
            'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "X"}',
            'data: {"type": "TOOL_CALL_START", "tool_call_id": "t2"}',
            "data: [DONE]",
        ]

        mock_callback = MagicMock()
        with patch("aidev_agent.core.tools.a2a_tools.local_backend.logger"):
            backend._run_subagent(  # noqa: SLF001
                mock_subagent, timeout_seconds=30, progress_callback=mock_callback
            )

        heartbeat_calls = [call for call in mock_callback.call_args_list if call[0][0] == "subagent.heartbeat"]
        # 3 parsed events (empty line skipped), 1 heartbeat per event
        assert len(heartbeat_calls) == 3


# ============== Test: 多轮交互文本提取 ==============


class TestLocalBackendMultiTurnTextExtraction:
    """LocalBackend 多轮交互文本提取：仅保留最后一轮 TEXT_MESSAGE 文本。"""

    def test_multi_turn_only_returns_final_answer(self) -> None:
        """多轮交互（含 TOOL_CALL）仅返回最后一轮文本，不包含中间思考。"""
        backend = LocalBackend()

        mock_subagent = MagicMock()
        mock_subagent.execute.return_value = [
            'data: {"type": "RUN_STARTED"}',
            'data: {"type": "TEXT_MESSAGE_START", "messageId": "msg_1"}',
            'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_1", "delta": "让我分析一下..."}',
            'data: {"type": "TEXT_MESSAGE_END", "messageId": "msg_1"}',
            'data: {"type": "TOOL_CALL_START", "toolCallId": "tc_1", "toolCallName": "analyze"}',
            'data: {"type": "TOOL_CALL_END", "toolCallId": "tc_1"}',
            'data: {"type": "TEXT_MESSAGE_START", "messageId": "msg_2"}',
            'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_2", "delta": "分析结果如下"}',
            'data: {"type": "TEXT_MESSAGE_END", "messageId": "msg_2"}',
            'data: {"type": "RUN_FINISHED"}',
            "data: [DONE]",
        ]

        result_text, events = backend._run_subagent(mock_subagent, timeout_seconds=30)  # noqa: SLF001

        assert result_text == "分析结果如下"
        assert "让我分析一下" not in result_text

    def test_single_turn_unchanged(self) -> None:
        """单轮交互（无 TOOL_CALL）行为与修改前一致。"""
        backend = LocalBackend()

        mock_subagent = MagicMock()
        mock_subagent.execute.return_value = [
            'data: {"type": "TEXT_MESSAGE_START", "messageId": "msg_1"}',
            'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_1", "delta": "直接回答"}',
            'data: {"type": "TEXT_MESSAGE_END", "messageId": "msg_1"}',
            'data: {"type": "RUN_FINISHED"}',
            "data: [DONE]",
        ]

        result_text, events = backend._run_subagent(mock_subagent, timeout_seconds=30)  # noqa: SLF001

        assert result_text == "直接回答"

    def test_tool_call_count_unaffected_by_text_message_start(self) -> None:
        """TEXT_MESSAGE_START 清空 text_parts 不影响 TOOL_CALL_START 统计。"""
        backend = LocalBackend()

        mock_subagent = MagicMock()
        mock_subagent.execute.return_value = [
            'data: {"type": "RUN_STARTED"}',
            'data: {"type": "TEXT_MESSAGE_START", "messageId": "msg_1"}',
            'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_1", "delta": "思考中..."}',
            'data: {"type": "TEXT_MESSAGE_END", "messageId": "msg_1"}',
            'data: {"type": "TOOL_CALL_START", "toolCallId": "tc_1", "toolCallName": "analyze"}',
            'data: {"type": "TOOL_CALL_END", "toolCallId": "tc_1"}',
            'data: {"type": "TEXT_MESSAGE_START", "messageId": "msg_2"}',
            'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_2", "delta": "最终回答"}',
            'data: {"type": "TEXT_MESSAGE_END", "messageId": "msg_2"}',
            'data: {"type": "RUN_FINISHED"}',
            "data: [DONE]",
        ]

        result_text, events = backend._run_subagent(mock_subagent, timeout_seconds=30)  # noqa: SLF001

        assert count_tool_calls(events) == 1

    def test_multi_turn_final_answer_across_deltas(self) -> None:
        """最终回答跨多个 TEXT_MESSAGE_CONTENT delta 时正确拼接。"""
        backend = LocalBackend()

        mock_subagent = MagicMock()
        mock_subagent.execute.return_value = [
            'data: {"type": "TEXT_MESSAGE_START", "messageId": "msg_1"}',
            'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_1", "delta": "思考中..."}',
            'data: {"type": "TEXT_MESSAGE_END", "messageId": "msg_1"}',
            'data: {"type": "TOOL_CALL_START", "toolCallId": "tc_1", "toolCallName": "search"}',
            'data: {"type": "TOOL_CALL_END", "toolCallId": "tc_1"}',
            'data: {"type": "TEXT_MESSAGE_START", "messageId": "msg_2"}',
            'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_2", "delta": "分析"}',
            'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_2", "delta": "完成"}',
            'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_2", "delta": "结论"}',
            'data: {"type": "TEXT_MESSAGE_END", "messageId": "msg_2"}',
            'data: {"type": "RUN_FINISHED"}',
            "data: [DONE]",
        ]

        result_text, events = backend._run_subagent(mock_subagent, timeout_seconds=30)  # noqa: SLF001

        assert result_text == "分析完成结论"
