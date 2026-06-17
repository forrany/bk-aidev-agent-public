# -*- coding: utf-8 -*-
"""A2A progress 工具函数单元测试。阶段 26 新增。

覆盖：
- build_enriched_result() 返回 AgentResult（D-02）
- count_tool_calls() 统计 TOOL_CALL_START
- sanitize_error_message() 脱敏
"""

from __future__ import annotations

import json

from aidev_agent.core.tools.a2a_tools.progress import (
    build_enriched_result,
    count_tool_calls,
    sanitize_error_message,
)
from aidev_agent.core.tools.a2a_tools.types import AgentResult, ExitReason


class TestBuildEnrichedResult:
    """build_enriched_result() 返回 AgentResult 验证（D-02 / D-07）。"""

    def test_returns_agent_result(self) -> None:
        """build_enriched_result 返回 AgentResult 实例而非 dict。"""
        r = build_enriched_result(status="completed", agent_name="test_agent")
        assert isinstance(r, AgentResult)

    def test_completed_status_fields(self) -> None:
        """completed 状态的 AgentResult 字段值正确。"""
        r = build_enriched_result(
            status="completed",
            agent_name="test_agent",
            summary="Done!",
            tool_calls=3,
            exit_reason=ExitReason.COMPLETED.value,
        )
        assert r.status == "completed"
        assert r.result == "Done!"
        assert r.error is None
        assert r.tool_calls == 3
        assert r.exit_reason == "completed"

    def test_failed_status_with_error_sanitization(self) -> None:
        """failed 状态时 error 字段被脱敏处理。"""
        r = build_enriched_result(
            status="failed",
            agent_name="test_agent",
            error="API error with sk-abc123def456ghi789jkl012mno345pqr678stu",
            exit_reason=ExitReason.BACKEND_ERROR.value,
        )
        assert r.status == "failed"
        assert "sk-" not in r.error
        assert "REDACTED" in r.error

    def test_default_field_values(self) -> None:
        """默认字段值：result="" / error=None / tool_calls=0 / agent_type=""。"""
        r = build_enriched_result(
            status="completed",
            agent_name="test_agent",
        )
        assert r.result == ""
        assert r.error is None
        assert r.tool_calls == 0
        assert r.exit_reason == "completed"
        assert r.agent_type == ""

    def test_agent_type_propagated(self) -> None:
        """agent_type 参数传递到 AgentResult 中。"""
        r = build_enriched_result(
            status="completed",
            agent_name="test_agent",
            agent_type="bkai",
        )
        assert r.agent_type == "bkai"
        d = r.model_dump()
        assert d["agent_type"] == "bkai"

    def test_agent_name_not_in_model(self) -> None:
        """agent_name 参数不包含在返回的 AgentResult 中（D-02）。"""
        r = build_enriched_result(
            status="completed",
            agent_name="my_agent",
        )
        d = r.model_dump()
        assert "agent_name" not in d

    def test_model_dump_json_excludes_none(self) -> None:
        """model_dump_json(exclude_unset=True) 排除 None 字段（D-03）。"""
        r = build_enriched_result(
            status="completed",
            agent_name="test_agent",
        )
        json_str = r.model_dump_json(exclude_unset=True)
        d = json.loads(json_str)
        # error 为 None（默认值，未设置）→ 被排除
        assert "error" not in d


class TestCountToolCalls:
    """count_tool_calls() 统计验证。"""

    def test_counts_tool_call_start_events(self) -> None:
        """统计 TOOL_CALL_START 事件次数。"""
        events = [
            {"type": "TEXT_MESSAGE_CONTENT", "delta": "hello"},
            {"type": "TOOL_CALL_START", "tool": "search"},
            {"type": "TOOL_CALL_START", "tool": "calculator"},
            {"type": "TEXT_MESSAGE_CONTENT", "delta": "world"},
        ]
        assert count_tool_calls(events) == 2

    def test_empty_events_returns_zero(self) -> None:
        """空事件列表返回 0。"""
        assert count_tool_calls([]) == 0

    def test_no_tool_calls_returns_zero(self) -> None:
        """无 TOOL_CALL_START 事件返回 0。"""
        events = [{"type": "TEXT_MESSAGE_CONTENT", "delta": "hi"}]
        assert count_tool_calls(events) == 0


class TestSanitizeErrorMessage:
    """sanitize_error_message() 脱敏验证。"""

    def test_redacts_api_key(self) -> None:
        """脱敏 sk- 前缀的 API key。"""
        msg = "Error: sk-abc123def456ghi789jkl012mno345"
        result = sanitize_error_message(msg)
        assert "sk-" not in result
        assert "REDACTED_API_KEY" in result

    def test_redacts_access_token(self) -> None:
        """脱敏 access_token 参数值。"""
        msg = "Error with access_token=my_secret_token_12345"
        result = sanitize_error_message(msg)
        assert "my_secret_token" not in result
        assert "REDACTED" in result

    def test_redacts_bearer_token(self) -> None:
        """脱敏 Bearer token。"""
        msg = "Auth failed: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        result = sanitize_error_message(msg)
        assert "eyJhbGci" not in result
        assert "REDACTED" in result

    def test_preserves_clean_message(self) -> None:
        """干净消息保持不变。"""
        msg = "Something went wrong"
        assert sanitize_error_message(msg) == msg
