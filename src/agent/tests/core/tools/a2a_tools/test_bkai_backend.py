# -*- coding: utf-8 -*-
"""BkaiBackend 远程后端单元测试。

覆盖 BkaiBackend 的：
- 零参构造和 Protocol 满足性
- 缺失 client/agent_code/session_code 时的 ValueError
- execute 正常调用路径（通过 SSE 流式请求）
- 异常捕获返回 failed 字典
- 流式调用验证
- prepare_session 方法
- caller_bk_app_code 传递
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from aidev_agent.core.tools.a2a_tools.types import AgentBackend, AgentBackendType, AgentSpec


def _make_mock_client() -> MagicMock:
    """创建 mock Client 实例。"""
    mock = MagicMock()
    # 设置 chat_completion Operation 的 path 属性（供流式请求构造 URL）
    mock.private_chat_completion.path = "/bk_plugin/private/agent/chat_completion/"
    mock.openapi_chat_completion.path = "/bk_plugin/openapi/agent/chat_completion/"
    # 让 operation.request() 失败，使流式请求走 session.post 降级路径
    mock.private_chat_completion.request.side_effect = AttributeError("stream via operation.request")
    mock.openapi_chat_completion.request.side_effect = AttributeError("stream via operation.request")
    return mock


def _setup_stream_response(
    mock_client: MagicMock,
    lines: list[str],
) -> MagicMock:
    """配置 mock Client 的 session.post 返回模拟 SSE 响应。

    Args:
        mock_client: mock Client 实例
        lines: SSE 行列表

    Returns:
        mock 响应对象
    """
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.iter_lines.return_value = lines
    mock_client.session.post.return_value = mock_resp
    mock_client._endpoint = "https://gateway.example.com/bp-test_001/prod"  # noqa: SLF001
    mock_client.session.headers = {}
    return mock_resp


def _make_bkai_spec(
    name: str = "test_agent",
    client: Any = None,
    **extra_params: Any,
) -> AgentSpec:
    """创建 BKAI 类型的 AgentSpec 测试辅助。"""
    params: dict[str, Any] = {}
    if client is not None:
        params["client"] = client
    params.update(extra_params)
    return AgentSpec(name=name, description="Test BKAI agent", backend_type=AgentBackendType.BKAI, params=params)


# ============== Test 1: 零参构造 ==============


class TestBkaiBackendInstantiate:
    """Test 1: BkaiBackend 类存在且零参构造成功。"""

    def test_bkai_backend_instantiate(self) -> None:
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        assert backend is not None


# ============== Test 2: AgentBackend Protocol ==============


class TestBkaiBackendSatisfiesProtocol:
    """Test 2: BkaiBackend 满足 AgentBackend Protocol (isinstance 检查)。"""

    def test_bkai_backend_satisfies_protocol(self) -> None:
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        assert isinstance(backend, AgentBackend)


# ============== Test 3: 缺失 client ==============


class TestBkaiBackendMissingClient:
    """Test 3: execute() 缺少 client 时抛出 ValueError。"""

    def test_missing_client_raises_value_error(self) -> None:
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        spec_no_client = AgentSpec(
            name="test_agent",
            description="Test",
            backend_type=AgentBackendType.BKAI,
            params={},
        )

        with pytest.raises(ValueError, match="Missing 'client' in spec.params"):
            backend.execute(spec_no_client, "hello", session_code="sess_test")

    def test_missing_session_code_raises_value_error(self) -> None:
        """execute() 缺少 session_code 时抛出 ValueError。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        spec = _make_bkai_spec(client=mock_client)

        with pytest.raises(ValueError, match="session_code must not be empty for BkaiBackend"):
            backend.execute(spec, "hello")


# ============== Test 5: execute 通过流式请求正常调用 ==============


class TestBkaiBackendExecuteSuccess:
    """Test 5: execute() 通过 SSE 流式请求正常调用并返回 completed。"""

    def test_execute_success_returns_completed(self) -> None:
        """execute() 流式请求返回 completed，result_text 来自 TEXT_MESSAGE_CONTENT 事件。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        _setup_stream_response(
            mock_client,
            [
                'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "子 Agent 返回的文本"}',
                "data: [DONE]",
            ],
        )

        spec = _make_bkai_spec(client=mock_client)

        result = backend.execute(spec, "请分析这段代码", session_code="sess_test")

        assert result.status == "completed"
        assert result.result == "子 Agent 返回的文本"

    def test_execute_calls_session_post_with_payload(self) -> None:
        """execute() 通过 client.session.post 发送流式请求，payload 包含 input 和 session_code。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        _setup_stream_response(
            mock_client,
            [
                'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "result"}',
                "data: [DONE]",
            ],
        )

        spec = _make_bkai_spec(client=mock_client)

        backend.execute(spec, "hello", session_code="sess_test")

        # 验证 session.post 被调用（operation.request 失败后走降级路径）
        mock_client.session.post.assert_called_once()
        post_call_kwargs = mock_client.session.post.call_args
        payload = post_call_kwargs.kwargs.get("json", {})
        assert payload.get("input") == "hello"
        assert payload.get("session_code") == "sess_test"

    def test_execute_payload_stream_true_and_persist_input(self) -> None:
        """execute() 请求体中 execute_kwargs.stream 为 True 且 persist_input 为 True。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        _setup_stream_response(
            mock_client,
            [
                'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "ok"}',
                "data: [DONE]",
            ],
        )

        spec = _make_bkai_spec(client=mock_client)

        backend.execute(spec, "hello", session_code="sess_test")

        post_call_kwargs = mock_client.session.post.call_args
        payload = post_call_kwargs.kwargs.get("json", {})
        assert payload.get("execute_kwargs", {}).get("stream") is True
        assert payload.get("execute_kwargs", {}).get("persist_input") is True
        assert payload.get("input") == "hello"
        assert payload.get("session_code") == "sess_test"

    def test_execute_counts_tool_calls(self) -> None:
        """execute() 统计 TOOL_CALL_START 事件数为 tool_calls。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        _setup_stream_response(
            mock_client,
            [
                'data: {"type": "TOOL_CALL_START", "tool_call_id": "t1"}',
                'data: {"type": "TOOL_CALL_START", "tool_call_id": "t2"}',
                'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "done"}',
                "data: [DONE]",
            ],
        )

        spec = _make_bkai_spec(client=mock_client)

        result = backend.execute(spec, "hello", session_code="sess_test")

        assert result.status == "completed"
        assert result.tool_calls == 2

    def test_execute_concatenates_multiple_text_deltas(self) -> None:
        """execute() 拼接多个 TEXT_MESSAGE_CONTENT 事件的 delta。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        _setup_stream_response(
            mock_client,
            [
                'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "Hello "}',
                'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "World"}',
                "data: [DONE]",
            ],
        )

        spec = _make_bkai_spec(client=mock_client)

        result = backend.execute(spec, "hello", session_code="sess_test")

        assert result.result == "Hello World"


# ============== Test 6: 异常捕获 ==============


class TestBkaiBackendExecuteException:
    """Test 6: execute() SSE 流式请求异常时返回 {status: "failed", agent_name, error}。"""

    def test_stream_error_event_returns_failed(self) -> None:
        """SSE 事件包含 type="error" 时，execute() 返回 failed。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        _setup_stream_response(
            mock_client,
            [
                'data: {"type": "error", "error": "Remote agent error"}',
                "data: [DONE]",
            ],
        )

        spec = _make_bkai_spec(client=mock_client)

        result = backend.execute(spec, "hello", session_code="sess_test")

        assert result.status == "failed"
        assert "Remote agent error" in result.error

    def test_http_exception_returns_failed(self) -> None:
        """HTTP 连接异常时，execute() 返回 failed。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        mock_client.session.post.side_effect = ConnectionError("网络连接失败")
        mock_client._endpoint = "https://gateway.example.com/bp-test_001/prod"  # noqa: SLF001
        mock_client.session.headers = {}

        spec = _make_bkai_spec(client=mock_client)

        result = backend.execute(spec, "hello", session_code="sess_test")

        assert result.status == "failed"
        assert "网络连接失败" in result.error

    def test_exception_does_not_propagate(self) -> None:
        """即使内部发生任何异常，execute() 也不应抛出。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        mock_client.session.post.side_effect = Exception("unexpected")
        mock_client._endpoint = "https://gateway.example.com/bp-test_001/prod"  # noqa: SLF001
        mock_client.session.headers = {}

        spec = _make_bkai_spec(client=mock_client)

        result = backend.execute(spec, "hello", session_code="sess_test")

        assert result.status == "failed"


# ============== Test 7: 无 graphs 层导入 ==============


class TestBkaiBackendNoGraphsImport:
    """Test 7: BkaiBackend 不导入 graphs 或 nodes 层（依赖方向合规）。"""

    def test_no_graphs_import(self) -> None:
        import aidev_agent.core.tools.a2a_tools.bkai_backend as mod

        with open(mod.__file__) as f:
            source = f.read()
        assert "from aidev_agent.core.graphs" not in source
        assert "from aidev_agent.core.nodes" not in source

    def test_no_services_agent_import(self) -> None:
        """BkaiBackend 不应导入 AgentInstanceFactory。"""
        import aidev_agent.core.tools.a2a_tools.bkai_backend as mod

        with open(mod.__file__) as f:
            source = f.read()
        assert "from aidev_agent.services.agent" not in source
        assert "AgentInstanceFactory" not in source

    def test_no_bk_agent_api_import(self) -> None:
        """BkaiBackend 不应导入 BkAgentApi（client 由外部注入）。"""
        import aidev_agent.core.tools.a2a_tools.bkai_backend as mod

        with open(mod.__file__) as f:
            source = f.read()
        assert "BkAgentApi" not in source

    def test_no_apigw_url_format_import(self) -> None:
        """BkaiBackend 不应导入 APIGW_URL_FORMAT（URL 由 Client 管理）。"""
        import aidev_agent.core.tools.a2a_tools.bkai_backend as mod

        with open(mod.__file__) as f:
            source = f.read()
        assert "APIGW_URL_FORMAT" not in source


# ============== Test: execute 委托关系 ==============


class TestBkaiBackendExecute:
    """execute() 方法测试：统一处理 task 和 member 模式。"""

    def test_execute_success(self) -> None:
        """execute() 正常返回 completed。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        _setup_stream_response(
            mock_client,
            [
                'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "chat result"}',
                "data: [DONE]",
            ],
        )

        spec = _make_bkai_spec(client=mock_client)

        result = backend.execute(spec, "hello", session_code="sess_test")

        assert result.status == "completed"
        assert result.result == "chat result"

    def test_execute_uses_stream_via_session_post(self) -> None:
        """execute() 通过 session.post 发送流式请求（operation.request 降级后）。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        _setup_stream_response(
            mock_client,
            [
                'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "direct result"}',
                "data: [DONE]",
            ],
        )

        spec = _make_bkai_spec(client=mock_client)

        result = backend.execute(spec, "hello", session_code="sess_test")

        assert result.status == "completed"
        assert result.result == "direct result"
        # 验证通过 session.post 降级路径发送请求
        mock_client.session.post.assert_called_once()


# ============== Test: prepare_session 方法 ==============


class TestBkaiBackendPrepareSession:
    """prepare_session() 方法单元测试。"""

    def test_prepare_session_calls_client_create_and_save(self) -> None:
        """prepare_session() 调用 client.create_session + client.save_session_content。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()

        backend.prepare_session(mock_client, "sess_abc", "test-session", "hello")

        mock_client.create_session.assert_called_once_with(
            data={"is_temporary": False, "session_code": "sess_abc", "session_name": "test-session"},
            headers={"X-BKAIDEV-USER": ""},
        )
        mock_client.save_session_content.assert_called_once_with(
            data={"session_code": "sess_abc", "role": "user", "content": "hello"}
        )

    def test_prepare_session_create_exception_logged(self) -> None:
        """prepare_session() create_session 异常时记录日志但不抛出。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        mock_client.create_session.side_effect = Exception("HTTP 503")

        # 不应抛出异常
        backend.prepare_session(mock_client, "sess_abc", "test-session", "hello")

        mock_client.save_session_content.assert_called_once()

    def test_prepare_session_save_exception_logged(self) -> None:
        """prepare_session() save_session_content 异常时记录日志但不抛出。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        mock_client.save_session_content.side_effect = Exception("HTTP 500")

        # 不应抛出异常
        backend.prepare_session(mock_client, "sess_abc", "test-session", "hello")

        mock_client.create_session.assert_called_once()

    def test_prepare_session_default_executor(self) -> None:
        """prepare_session() 不传 executor 时仍然正常调用 client API。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()

        backend.prepare_session(mock_client, "sess_abc", "test-session", "hello")

        mock_client.create_session.assert_called_once_with(
            data={"is_temporary": False, "session_code": "sess_abc", "session_name": "test-session"},
            headers={"X-BKAIDEV-USER": ""},
        )
        mock_client.save_session_content.assert_called_once_with(
            data={"session_code": "sess_abc", "role": "user", "content": "hello"}
        )


# ============== Test: execute member 模式增强 ==============


class TestBkaiBackendExecuteMemberMode:
    """execute() member 模式增强测试（session_code 非空，should_prepare_session=True）。"""

    def test_execute_with_should_prepare_session_calls_prepare(self) -> None:
        """should_prepare_session=True 时调用 prepare_session 准备远端 session。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        _setup_stream_response(
            mock_client,
            [
                'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "result"}',
                "data: [DONE]",
            ],
        )

        spec = _make_bkai_spec(client=mock_client, should_prepare_session=True)

        with patch.object(backend, "prepare_session") as mock_prepare:
            backend.execute(spec, "hello user", session_code="sess_123")

        mock_prepare.assert_called_once()

    def test_execute_without_should_prepare_session_skips_prepare(self) -> None:
        """should_prepare_session=False（默认）时不调用 prepare_session。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        _setup_stream_response(
            mock_client,
            [
                'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "ok"}',
                "data: [DONE]",
            ],
        )

        spec = _make_bkai_spec(client=mock_client)

        with patch.object(backend, "prepare_session") as mock_prepare:
            result = backend.execute(spec, "hello", session_code="sess_456")

        assert result.status == "completed"
        mock_prepare.assert_not_called()

    def test_execute_result_contains_text(self) -> None:
        """execute() 返回结果中包含流式文本。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        _setup_stream_response(
            mock_client,
            [
                'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "assistant reply"}',
                "data: [DONE]",
            ],
        )

        spec = _make_bkai_spec(client=mock_client, should_prepare_session=True)

        with patch.object(backend, "prepare_session"):
            result = backend.execute(spec, "hello", session_code="sess_123")

        assert result.status == "completed"
        assert result.result == "assistant reply"


# ============== Test: _extract_execute_kwargs ==============


class TestBkaiBackendExtractExecuteKwargs:
    """_extract_execute_kwargs() 辅助方法测试。"""

    def test_extract_execute_kwargs_with_execute_kwargs_object(self) -> None:
        """_extract_execute_kwargs() 从 config 中的 ExecuteKwargs Pydantic 对象序列化。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend
        from aidev_agent.pydantic_models import ExecuteKwargs

        backend = BkaiBackend()

        ek = ExecuteKwargs(executor="user1", session_code="sess_abc", caller_bk_app_code="app_123")
        config = {"configurable": {"execute_kwargs": ek}}

        result = backend._extract_execute_kwargs(
            config, session_code="sub_sess", stream=True, caller_bk_app_code="caller_app"
        )  # noqa: SLF001

        assert result["stream"] is True
        assert result["persist_input"] is True
        assert result["session_code"] == "sub_sess"
        # caller_bk_app_code 通过参数传入，覆盖原始 ExecuteKwargs 中的值
        assert result["caller_bk_app_code"] == "caller_app"
        # executor 从原始 ExecuteKwargs 继承
        assert result["executor"] == "user1"

    def test_extract_execute_kwargs_no_config(self) -> None:
        """_extract_execute_kwargs() 无 config 时只设置覆盖字段。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()

        result = backend._extract_execute_kwargs(None, session_code="sub_sess")  # noqa: SLF001

        assert result["stream"] is True
        assert result["persist_input"] is True
        assert result["session_code"] == "sub_sess"

    def test_extract_execute_kwargs_non_execute_kwargs_object_ignored(self) -> None:
        """_extract_execute_kwargs() config 中的 execute_kwargs 非 ExecuteKwargs 实例时忽略。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()

        # 非 ExecuteKwargs 对象（如 dict）不会被序列化
        config = {"configurable": {"execute_kwargs": {"executor": "user1"}}}

        result = backend._extract_execute_kwargs(config, session_code="sub_sess")  # noqa: SLF001

        # 只包含覆盖字段，原始 execute_kwargs 被忽略
        assert result["stream"] is True
        assert result["persist_input"] is True
        assert result["session_code"] == "sub_sess"
        assert "executor" not in result

    def test_extract_execute_kwargs_caller_bk_app_code(self) -> None:
        """_extract_execute_kwargs() 将 caller_bk_app_code 参数写入 execute_kwargs。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()

        result = backend._extract_execute_kwargs(None, session_code="sub_sess", caller_bk_app_code="my_app_code")  # noqa: SLF001

        assert result["caller_bk_app_code"] == "my_app_code"

    def test_extract_execute_kwargs_no_caller_bk_app_code(self) -> None:
        """_extract_execute_kwargs() 无 caller_bk_app_code 时不设置该字段。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()

        result = backend._extract_execute_kwargs(None, session_code="sub_sess")  # noqa: SLF001

        assert "caller_bk_app_code" not in result


# ============== Test: Client 从 spec.params 注入 ==============


class TestBkaiBackendClientInjection:
    """BkaiBackend 从 spec.params["client"] 获取 Client，不负责构造。"""

    def test_execute_uses_injected_client(self) -> None:
        """execute() 使用注入的 Client。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        _setup_stream_response(
            mock_client,
            [
                'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "ok"}',
                "data: [DONE]",
            ],
        )

        spec = _make_bkai_spec(client=mock_client)
        result = backend.execute(spec, "hello", session_code="sess_test")

        assert result.status == "completed"
        mock_client.session.post.assert_called_once()


# ============== Test 16: BkaiBackend 超时诊断 ==============


class TestBkaiBackendTimeoutWarning:
    """Test 16: execute() 超时时输出结构化 logger.warning（D-01）。"""

    def test_timeout_emits_warning_with_correct_fields(self) -> None:
        """当 execute() 捕获到包含 "timeout" 的异常时，调用 logger.warning 输出结构化信息。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        # 模拟 session.post 抛出超时异常
        mock_client.session.post.side_effect = Exception("Request timeout after 30s")
        mock_client._endpoint = "https://gateway.example.com/bp-test_001/prod"  # noqa: SLF001
        mock_client.session.headers = {}
        spec = _make_bkai_spec(name="timeout_agent", client=mock_client)
        spec.timeout_seconds = 30

        with patch("aidev_agent.core.tools.a2a_tools.bkai_backend.logger") as mock_logger:
            result = backend.execute(spec, "hello", session_code="sess_test")

        assert result.status == "failed"
        assert result.exit_reason == "timeout"
        assert mock_logger.warning.called

        # 查找超时相关的 warning 调用
        timeout_calls = [call for call in mock_logger.warning.call_args_list if "A2A subagent timeout" in str(call[0])]
        assert len(timeout_calls) >= 1

    def test_non_timeout_error_no_timeout_warning(self) -> None:
        """非超时异常（如凭据错误）不输出超时警告。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        mock_client.session.post.side_effect = Exception("401 Unauthorized: credential error")
        mock_client._endpoint = "https://gateway.example.com/bp-test_001/prod"  # noqa: SLF001
        mock_client.session.headers = {}
        spec = _make_bkai_spec(name="cred_error_agent", client=mock_client)

        with patch("aidev_agent.core.tools.a2a_tools.bkai_backend.logger") as mock_logger:
            result = backend.execute(spec, "hello", session_code="sess_test")

        assert result.status == "failed"
        assert result.exit_reason == "credential_error"

        timeout_warning_found = any("A2A subagent timeout" in str(call) for call in mock_logger.warning.call_args_list)
        assert not timeout_warning_found

    def test_non_empty_session_code_in_timeout_warning(self) -> None:
        """非空 session_code 显示在超时警告中。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        mock_client.session.post.side_effect = Exception("timeout")
        mock_client._endpoint = "https://gateway.example.com/bp-test_001/prod"  # noqa: SLF001
        mock_client.session.headers = {}
        spec = _make_bkai_spec(name="member_agent", client=mock_client)

        with patch("aidev_agent.core.tools.a2a_tools.bkai_backend.logger") as mock_logger:
            backend.execute(spec, "hello", session_code="abc123")

        timeout_calls = [call for call in mock_logger.warning.call_args_list if "A2A subagent timeout" in str(call[0])]
        assert len(timeout_calls) >= 1
        call_args = timeout_calls[0][0]
        assert call_args[-1] == "abc123"


# ============== Test 17: BkaiBackend 心跳发送 ==============


class TestBkaiBackendHeartbeat:
    """Test 17: 流式方法中发送 subagent.heartbeat 事件（D-04）。"""

    def test_heartbeat_events_sent_with_callback(self) -> None:
        """当提供 progress_callback 时，_chat_completion_stream_via_client 发送心跳。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()

        # 模拟 HTTP 响应的 iter_lines
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "Hello"}',
            'data: {"type": "TOOL_CALL_START", "tool_call_id": "t1"}',
            "data: [DONE]",
        ]
        mock_client.session.post.return_value = mock_resp

        spec = _make_bkai_spec(client=mock_client)
        mock_callback = MagicMock()

        with patch("aidev_agent.core.tools.a2a_tools.bkai_backend.logger"):
            events = list(
                backend._chat_completion_stream_via_client(  # noqa: SLF001
                    mock_client,
                    spec,
                    "hello",
                    base_execute_kwargs={},
                    progress_callback=mock_callback,
                )
            )

        assert len(events) == 2  # DONE is not yielded

        heartbeat_calls = [call for call in mock_callback.call_args_list if call[0][0] == "subagent.heartbeat"]
        assert len(heartbeat_calls) == 2

        first_kwargs = heartbeat_calls[0][1]
        assert "elapsed_seconds" in first_kwargs
        assert "tool_count" in first_kwargs
        assert "iteration" in first_kwargs

    def test_no_heartbeat_when_callback_none(self) -> None:
        """当 progress_callback 为 None 时，无心跳事件。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "Hello"}',
            "data: [DONE]",
        ]
        mock_client.session.post.return_value = mock_resp

        spec = _make_bkai_spec(client=mock_client)

        with patch("aidev_agent.core.tools.a2a_tools.bkai_backend.logger"):
            events = list(
                backend._chat_completion_stream_via_client(  # noqa: SLF001
                    mock_client,
                    spec,
                    "hello",
                    base_execute_kwargs={},
                )
            )

        assert len(events) == 1
        # 无 callback → 无心跳，不报错

    def test_heartbeat_once_per_sse_event(self) -> None:
        """每个解析的 SSE 事件触发一次心跳。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            'data: {"type": "RUN_STARTED"}',
            "",
            'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "X"}',
            'data: {"type": "TOOL_CALL_START", "tool_call_id": "t2"}',
            "data: [DONE]",
        ]
        mock_client.session.post.return_value = mock_resp

        spec = _make_bkai_spec(client=mock_client)
        mock_callback = MagicMock()

        with patch("aidev_agent.core.tools.a2a_tools.bkai_backend.logger"):
            list(
                backend._chat_completion_stream_via_client(  # noqa: SLF001
                    mock_client,
                    spec,
                    "hello",
                    base_execute_kwargs={},
                    progress_callback=mock_callback,
                )
            )

        heartbeat_calls = [call for call in mock_callback.call_args_list if call[0][0] == "subagent.heartbeat"]
        # 3 parsed events (empty line skipped) → 3 heartbeats
        assert len(heartbeat_calls) == 3

    def test_heartbeat_includes_tool_count(self) -> None:
        """心跳中的 tool_count 正确反映 TOOL_CALL_START 事件数。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            'data: {"type": "TOOL_CALL_START", "tool_call_id": "t_a"}',
            'data: {"type": "TOOL_CALL_START", "tool_call_id": "t_b"}',
            'data: {"type": "TEXT_MESSAGE_CONTENT", "delta": "Done"}',
            "data: [DONE]",
        ]
        mock_client.session.post.return_value = mock_resp

        spec = _make_bkai_spec(client=mock_client)
        mock_callback = MagicMock()

        with patch("aidev_agent.core.tools.a2a_tools.bkai_backend.logger"):
            list(
                backend._chat_completion_stream_via_client(  # noqa: SLF001
                    mock_client,
                    spec,
                    "hello",
                    base_execute_kwargs={},
                    progress_callback=mock_callback,
                )
            )

        heartbeat_calls = [call for call in mock_callback.call_args_list if call[0][0] == "subagent.heartbeat"]
        # 最后一条心跳的 tool_count 应为 2
        last_kwargs = heartbeat_calls[-1][1]
        assert last_kwargs["tool_count"] == 2
        assert last_kwargs["iteration"] == 3


# ============== Test: 多轮交互文本提取 ==============


class TestBkaiBackendMultiTurnTextExtraction:
    """BkaiBackend 多轮交互文本提取：仅保留最后一轮 TEXT_MESSAGE 文本。"""

    def test_multi_turn_only_returns_final_answer(self) -> None:
        """多轮交互（含 TOOL_CALL）仅返回最后一轮文本，不包含中间思考。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        _setup_stream_response(
            mock_client,
            [
                'data: {"type": "RUN_STARTED"}',
                'data: {"type": "TEXT_MESSAGE_START", "messageId": "msg_1"}',
                'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_1", "delta": "让我查一下天气信息..."}',
                'data: {"type": "TEXT_MESSAGE_END", "messageId": "msg_1"}',
                'data: {"type": "TOOL_CALL_START", "toolCallId": "tc_1", "toolCallName": "get_weather"}',
                'data: {"type": "TOOL_CALL_ARGS", "toolCallId": "tc_1", "delta": "{\\"location\\": \\"广州\\"}"}',
                'data: {"type": "TOOL_CALL_END", "toolCallId": "tc_1"}',
                'data: {"type": "TEXT_MESSAGE_START", "messageId": "msg_2"}',
                'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_2", "delta": "广州今天多云，温度25度。"}',
                'data: {"type": "TEXT_MESSAGE_END", "messageId": "msg_2"}',
                'data: {"type": "RUN_FINISHED"}',
                "data: [DONE]",
            ],
        )

        spec = _make_bkai_spec(client=mock_client)
        result = backend.execute(spec, "广州天气", session_code="sess_test")

        assert "广州今天多云" in result.result
        assert "让我查一下" not in result.result

    def test_single_turn_unchanged(self) -> None:
        """单轮交互（无 TOOL_CALL）行为与修改前一致。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        _setup_stream_response(
            mock_client,
            [
                'data: {"type": "TEXT_MESSAGE_START", "messageId": "msg_1"}',
                'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_1", "delta": "直接回答"}',
                'data: {"type": "TEXT_MESSAGE_END", "messageId": "msg_1"}',
                'data: {"type": "RUN_FINISHED"}',
                "data: [DONE]",
            ],
        )

        spec = _make_bkai_spec(client=mock_client)
        result = backend.execute(spec, "hello", session_code="sess_test")

        assert result.result == "直接回答"

    def test_tool_call_count_unaffected_by_text_message_start(self) -> None:
        """TEXT_MESSAGE_START 清空 text_parts 不影响 TOOL_CALL_START 统计。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        _setup_stream_response(
            mock_client,
            [
                'data: {"type": "RUN_STARTED"}',
                'data: {"type": "TEXT_MESSAGE_START", "messageId": "msg_1"}',
                'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_1", "delta": "思考中..."}',
                'data: {"type": "TEXT_MESSAGE_END", "messageId": "msg_1"}',
                'data: {"type": "TOOL_CALL_START", "toolCallId": "tc_1", "toolCallName": "search"}',
                'data: {"type": "TOOL_CALL_END", "toolCallId": "tc_1"}',
                'data: {"type": "TOOL_CALL_START", "toolCallId": "tc_2", "toolCallName": "lookup"}',
                'data: {"type": "TOOL_CALL_END", "toolCallId": "tc_2"}',
                'data: {"type": "TEXT_MESSAGE_START", "messageId": "msg_2"}',
                'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_2", "delta": "最终回答"}',
                'data: {"type": "TEXT_MESSAGE_END", "messageId": "msg_2"}',
                'data: {"type": "RUN_FINISHED"}',
                "data: [DONE]",
            ],
        )

        spec = _make_bkai_spec(client=mock_client)
        result = backend.execute(spec, "hello", session_code="sess_test")

        assert result.tool_calls == 2

    def test_multi_turn_final_answer_across_deltas(self) -> None:
        """最终回答跨多个 TEXT_MESSAGE_CONTENT delta 时正确拼接。"""
        from aidev_agent.core.tools.a2a_tools.bkai_backend import BkaiBackend

        backend = BkaiBackend()
        mock_client = _make_mock_client()
        _setup_stream_response(
            mock_client,
            [
                'data: {"type": "TEXT_MESSAGE_START", "messageId": "msg_1"}',
                'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_1", "delta": "思考中..."}',
                'data: {"type": "TEXT_MESSAGE_END", "messageId": "msg_1"}',
                'data: {"type": "TOOL_CALL_START", "toolCallId": "tc_1", "toolCallName": "search"}',
                'data: {"type": "TOOL_CALL_END", "toolCallId": "tc_1"}',
                'data: {"type": "TEXT_MESSAGE_START", "messageId": "msg_2"}',
                'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_2", "delta": "广州"}',
                'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_2", "delta": "今天"}',
                'data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "msg_2", "delta": "多云"}',
                'data: {"type": "TEXT_MESSAGE_END", "messageId": "msg_2"}',
                'data: {"type": "RUN_FINISHED"}',
                "data: [DONE]",
            ],
        )

        spec = _make_bkai_spec(client=mock_client)
        result = backend.execute(spec, "广州天气", session_code="sess_test")

        assert result.result == "广州今天多云"
