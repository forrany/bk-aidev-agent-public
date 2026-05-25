# -*- coding: utf-8 -*-
"""标准运维插件核心契约：Session 轮询 + Chat / Flow 编排。"""

import sys
from unittest.mock import ANY, MagicMock, patch

import pytest

try:
    import django  # noqa: F401

    _has_django = True
except ImportError:
    _has_django = False

if "pkg_resources" not in sys.modules:
    sys.modules["pkg_resources"] = MagicMock()

_bk_plugin_framework = MagicMock()
_bk_plugin_framework.kit.decorators.inject_user_token = lambda func: func
sys.modules.setdefault("bk_plugin_framework", _bk_plugin_framework)
sys.modules.setdefault("bk_plugin_framework.kit", _bk_plugin_framework.kit)
sys.modules.setdefault("bk_plugin_framework.kit.decorators", _bk_plugin_framework.kit.decorators)

from aidev_agent.enums import AgentType, ChatContentStatus, PromptRole, SessionsStatus  # noqa: E402
from aidev_bkplugin.enums import PluginPollTaskState  # noqa: E402
from aidev_bkplugin.services.agent_session import SessionManager  # noqa: E402


class _FakeApi:
    def __init__(self, contents, status=SessionsStatus.FINISHED.value):
        self._contents = contents
        self._status = status

    def retrieve_chat_session(self, *, path_params, headers):
        return {"data": {"status": self._status}}

    def get_chat_session_contents(self, *, params, headers):
        return {"data": self._contents}


def _patch_poll_client(monkeypatch, contents, status):
    monkeypatch.setattr(SessionManager, "_client", lambda self: type("C", (), {"api": _FakeApi(contents, status)})())


@pytest.mark.parametrize(
    "status,contents,turn_id,expected",
    [
        (SessionsStatus.RUNNING.value, [], "", (PluginPollTaskState.RUNNING, "")),
        (
            SessionsStatus.FINISHED.value,
            [{"role": PromptRole.ASSISTANT.value, "status": ChatContentStatus.COMPLETE.value, "content": "done"}],
            "",
            (PluginPollTaskState.SUCCESS, "done"),
        ),
        (
            SessionsStatus.FINISHED.value,
            [
                {
                    "role": PromptRole.AI.value,
                    "status": ChatContentStatus.SUCCESS.value,
                    "content": "old",
                    "property": {"turn_id": "t-old"},
                },
                {
                    "role": PromptRole.AI.value,
                    "status": ChatContentStatus.SUCCESS.value,
                    "content": "hit",
                    "property": {"turn_id": "t-now"},
                },
                {
                    "role": PromptRole.AI.value,
                    "status": ChatContentStatus.SUCCESS.value,
                    "content": "other",
                    "property": {"turn_id": "t-x"},
                },
            ],
            "t-now",
            (PluginPollTaskState.SUCCESS, "hit"),
        ),
        (
            SessionsStatus.FINISHED.value,
            [{"role": PromptRole.USER.value, "status": ChatContentStatus.COMPLETE.value, "content": "hi"}],
            "",
            (PluginPollTaskState.SUCCESS, ""),
        ),
    ],
)
def test_poll_task_state(status, contents, turn_id, expected, monkeypatch):
    _patch_poll_client(monkeypatch, contents, status)
    assert SessionManager("alice").poll_task_state("sc", turn_id=turn_id) == expected


def test_prepare_session_turn_uses_user_content_turn_id(monkeypatch):
    sm = SessionManager("alice")
    monkeypatch.setattr(sm, "get_or_create_by_thread_id", lambda tid: "sc-1")
    saved: list[tuple[str, str]] = []

    def _save(session_code, role, content, **kw):
        saved.append((role, kw.get("turn_id", "")))
        return {"property": {"turn_id": kw.get("turn_id") or "generated-turn"}}

    monkeypatch.setattr(sm, "save_content", _save)

    assert sm.prepare_session_turn("thread-a", input_text="q") == ("sc-1", "generated-turn")
    assert saved == [(PromptRole.USER.value, "")]


@pytest.mark.skipif(not _has_django, reason="Django and plugin env required")
class TestBkpluginExecution:
    @pytest.mark.parametrize("agent_type", [AgentType.FLOW, AgentType.CHAT])
    def test_stream_execute_returns_async_storage(self, agent_type):
        from aidev_bkplugin.services.agent_bkplugin import BkpluginAgentRunner

        with (
            patch("aidev_bkplugin.services.agent_bkplugin.AgentConfigFetcher.get_info") as mock_info,
            patch("aidev_bkplugin.services.agent_bkplugin.SessionManager") as mock_sm_cls,
        ):
            mock_info.return_value = {"agent_type": agent_type.value}
            mock_sm = mock_sm_cls.return_value
            mock_sm.prepare_session_turn.return_value = ("sess-1", "turn-1")
            agent = BkpluginAgentRunner.create(
                chat_history=[],
                execute_kwargs={"session_code": "thread-1"},
                input_text="hi",
                username="alice",
                stream=True,
            )
            agent.start_background_task = MagicMock()
            result = agent.execute()

        mock_sm.prepare_session_turn.assert_called_once_with("thread-1", input_text="hi", turn_id=ANY)
        assert result.is_async and result.storage["turn_id"] and result.storage["agent_type"] == agent_type.value
        agent.start_background_task.assert_called_once()
        assert agent.start_background_task.call_args.args[1]["turn_id"] == "turn-1"
        assert agent.start_background_task.call_args.args[1]["stream"] is True

    def test_start_background_task_uses_celery_delay(self):
        from aidev_bkplugin.services.agent_bkplugin import BkpluginChat

        task = MagicMock()
        agent = BkpluginChat(
            chat_history=[],
            execute_kwargs={},
            username="alice",
        )
        with patch("aidev_bkplugin.tasks.run_bkplugin_background_agent_task", task):
            agent.start_background_task(
                "sess-1",
                {"turn_id": "turn-1"},
                AgentType.CHAT,
                chat_context=[{"role": "user", "content": "hi"}],
            )

        task.delay.assert_called_once_with(
            session_code="sess-1",
            execute_kwargs={"turn_id": "turn-1"},
            username="alice",
            agent_type_value=AgentType.CHAT.value,
            chat_context=[{"role": "user", "content": "hi"}],
        )

    def test_serial_chat_execute(self):
        from aidev_bkplugin.services.agent_bkplugin import BkpluginAgentRunner

        with (
            patch("aidev_bkplugin.services.agent_bkplugin.AgentConfigFetcher.get_info") as mock_info,
            patch("aidev_bkplugin.services.agent_bkplugin.SessionManager") as mock_sm_cls,
            patch("aidev_bkplugin.services.agent_bkplugin.build_chat_agent_for_session") as mock_build,
        ):
            mock_info.return_value = {"agent_type": AgentType.CHAT.value}
            mock_sm = mock_sm_cls.return_value
            mock_sm.prepare_session_turn.return_value = ("sess-chat", "turn-chat")
            mock_agent = MagicMock(execute=MagicMock(return_value={"choices": [{"delta": {"content": "ok"}}]}))
            mock_build.return_value = mock_agent
            agent = BkpluginAgentRunner.create(
                chat_history=[{"role": "user", "content": "prev"}],
                execute_kwargs={"session_code": "thread-1"},
                input_text="hi",
                username="alice",
                stream=False,
            )
            result = agent.execute()

        assert not result.is_async and result.result["choices"][0]["delta"]["content"] == "ok"
        assert [x["content"] for x in mock_build.call_args.kwargs["chat_context"]] == ["prev", "hi"]
        assert mock_build.call_args.kwargs["turn_id"] == "turn-chat"
        mock_sm.save_ai_response.assert_called_once_with("sess-chat", result.result, turn_id=ANY)

    def test_serial_flow_execute(self):
        from aidev_bkplugin.enums import PluginPollTaskState
        from aidev_bkplugin.services.agent_bkplugin import BkpluginAgentRunner

        with (
            patch("aidev_bkplugin.services.agent_bkplugin.AgentConfigFetcher.get_info") as mock_info,
            patch("aidev_bkplugin.services.agent_bkplugin.BkpluginFlow._run_flow") as mock_run_flow,
            patch("aidev_bkplugin.services.agent_bkplugin.SessionManager") as mock_sm_cls,
        ):
            mock_info.return_value = {"agent_type": AgentType.FLOW.value}
            mock_sm = mock_sm_cls.return_value
            mock_sm.prepare_session_turn.return_value = ("sess-flow", "turn-flow")
            mock_sm.poll_task_state.return_value = (PluginPollTaskState.SUCCESS, "flow out")
            agent = BkpluginAgentRunner.create(
                chat_history=[],
                execute_kwargs={"session_code": "thread-1"},
                input_text="hi",
                username="alice",
                stream=False,
            )
            result = agent.execute()

        mock_sm.poll_task_state.assert_called_once_with("sess-flow", turn_id=ANY)
        assert mock_run_flow.call_args.args[1]["turn_id"] == "turn-flow"
        assert not result.is_async and result.result == "flow out"
