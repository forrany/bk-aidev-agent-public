# -*- coding: utf-8 -*-
"""``AgentBuilder`` OO 入口：装配 event_handler / checkpointer，确保 thread 路径顺序正确。"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _patch_factories():
    return (
        patch("aidev_bkplugin.services.agent_builder.AgentInstanceFactory.build_agent"),
        patch("aidev_bkplugin.services.agent_builder.common_agent_factory.get"),
    )


def test_by_session_code_passes_version_and_attaches_event_handler():
    from aidev_bkplugin.services.agent_builder import AgentBuilder

    build_p, factory_p = _patch_factories()
    with build_p as mock_build, factory_p as mock_factory:
        mock_factory.return_value = MagicMock(name="agent_cls")
        mock_build.return_value = MagicMock(name="agent")
        AgentBuilder(username="alice").by_session_code("sc-1", version="v2")

    kwargs = mock_build.call_args.kwargs
    assert kwargs["session_code"] == "sc-1"
    assert kwargs["version"] == "v2"
    assert kwargs["event_handler"] is not None
    assert kwargs["checkpointer"] is not None


def test_by_thread_id_with_chat_history_persists_then_builds():
    """Builder 走 SessionManager 实例方法路径：先 ensure session → 写入 history → 装配 agent。"""
    from aidev_bkplugin.services.agent_builder import AgentBuilder

    history = [
        SimpleNamespace(role="user", content="hello"),
        SimpleNamespace(role="assistant", content="hi"),
    ]
    sm = MagicMock(name="session_manager")
    sm.get_or_create_by_thread_id.return_value = "session-xyz"

    build_p, factory_p = _patch_factories()
    with build_p as mock_build, factory_p as mock_factory:
        mock_factory.return_value = MagicMock()
        mock_build.return_value = MagicMock()
        builder = AgentBuilder(username="alice", session_manager=sm)
        agent_instance, session_code = builder.by_thread_id_with_chat_history("t-1", history)

    assert session_code == "session-xyz"
    assert agent_instance is mock_build.return_value
    sm.get_or_create_by_thread_id.assert_called_once_with("t-1")
    sm.save_chat_history.assert_called_once_with("session-xyz", history)


def test_by_thread_id_skips_save_when_save_content_false():
    from aidev_bkplugin.services.agent_builder import AgentBuilder

    sm = MagicMock()
    sm.get_or_create_by_thread_id.return_value = "session-xyz"

    build_p, factory_p = _patch_factories()
    with build_p as mock_build, factory_p as mock_factory:
        mock_factory.return_value = MagicMock()
        mock_build.return_value = MagicMock()
        AgentBuilder(username="alice", session_manager=sm).by_thread_id("t-1", "hello", save_content=False)

    sm.save_content.assert_not_called()


def test_by_chat_history_prepends_role_prompts(monkeypatch):
    from aidev_agent.pydantic_models import ChatPrompt
    from aidev_bkplugin.services.agent_builder import AgentBuilder

    role_prompts = [ChatPrompt(role="system", content="role-x")]
    fake_fetcher = MagicMock()
    fake_fetcher.get_role_info.return_value = role_prompts
    monkeypatch.setattr("aidev_bkplugin.services.agent_builder.AgentConfigFetcher", fake_fetcher)

    build_p, factory_p = _patch_factories()
    history = [ChatPrompt(role="user", content="hi")]
    with build_p as mock_build, factory_p as mock_factory:
        mock_factory.return_value = MagicMock()
        mock_build.return_value = MagicMock()
        AgentBuilder(username="alice").by_chat_history(history)

    session_data = mock_build.call_args.kwargs["session_context_data"]
    assert [item["role"] for item in session_data] == ["system", "user"]
