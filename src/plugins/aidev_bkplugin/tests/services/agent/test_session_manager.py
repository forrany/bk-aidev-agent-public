# -*- coding: utf-8 -*-
"""``SessionManager`` 行为契约：哈希稳定性、404 → create 分支、save_content kwargs 透传。"""

from types import SimpleNamespace

import pytest
from bkapi_client_core.exceptions import HTTPResponseError


@pytest.fixture
def session_manager():
    from aidev_bkplugin.services.agent_session import SessionManager

    return SessionManager(username="alice", agent_code="bk-aidev")


@pytest.mark.parametrize(
    "username, agent_code, thread_id, expected_len",
    [
        ("alice", "bk-aidev", "t-1", 32),
        ("bob", "bk-aidev", "t-1", 32),
    ],
)
def test_generate_session_code_is_stable_md5(username, agent_code, thread_id, expected_len):
    from aidev_bkplugin.services.agent_session import SessionManager

    code = SessionManager.generate_session_code(username, agent_code, thread_id)
    assert len(code) == expected_len
    assert SessionManager.generate_session_code(username, agent_code, thread_id) == code


def _make_response_error(status_code: int) -> HTTPResponseError:
    """构造带 ``response.status_code`` 属性的 HTTPResponseError；``response_status_code`` 由父类 property 推导。"""
    return HTTPResponseError("err", response=SimpleNamespace(status_code=status_code, headers={}))


def test_get_or_create_returns_existing_session(session_manager, mock_plugin_rm_client):
    mock_plugin_rm_client.api.retrieve_chat_session.return_value = {"data": {"session_code": "abc"}}

    result = session_manager.get_or_create_by_thread_id("t-1")

    assert result == session_manager.generate_session_code("alice", "bk-aidev", "t-1")
    mock_plugin_rm_client.api.create_chat_session.assert_not_called()


def test_get_or_create_falls_back_to_create_on_404(session_manager, mock_plugin_rm_client):
    mock_plugin_rm_client.api.retrieve_chat_session.side_effect = _make_response_error(404)
    mock_plugin_rm_client.api.create_chat_session.return_value = {"data": {}}

    session_manager.get_or_create_by_thread_id("thread-12345678abc")

    create_call = mock_plugin_rm_client.api.create_chat_session.call_args
    assert create_call.kwargs["json"]["session_name"] == "Thread-thread-1"


def test_get_or_create_propagates_non_404(session_manager, mock_plugin_rm_client):
    mock_plugin_rm_client.api.retrieve_chat_session.side_effect = _make_response_error(500)
    with pytest.raises(HTTPResponseError):
        session_manager.get_or_create_by_thread_id("t-1")


@pytest.mark.parametrize(
    "extra, status, expect_extra",
    [
        (None, "success", False),
        ({"k": "v"}, "error", True),
    ],
)
def test_save_content_passes_extra_and_status(session_manager, mock_plugin_rm_client, extra, status, expect_extra):
    mock_plugin_rm_client.api.create_chat_session_content.return_value = {"data": {"id": 1}}

    session_manager.save_content("sc-1", "user", "hello", extra=extra, status=status, turn_id="turn-1")

    payload = mock_plugin_rm_client.api.create_chat_session_content.call_args.kwargs["json"]
    assert payload["status"] == status
    assert ("extra" in payload) is expect_extra
    assert payload["property"]["turn_id"] == "turn-1"


def test_save_content_generates_turn_id_for_user(session_manager, mock_plugin_rm_client):
    mock_plugin_rm_client.api.create_chat_session_content.return_value = {"data": {"id": 1}}

    saved = session_manager.save_content("sc-1", "user", "hello")

    payload = mock_plugin_rm_client.api.create_chat_session_content.call_args.kwargs["json"]
    assert payload["property"]["turn_id"]
    assert saved["property"]["turn_id"] == payload["property"]["turn_id"]


def test_prepare_session_turn_inherits_user_turn_id_without_input(session_manager, mock_plugin_rm_client, monkeypatch):
    mock_plugin_rm_client.api.retrieve_chat_session.return_value = {"data": {"session_code": "sc-1"}}
    monkeypatch.setattr(
        session_manager,
        "list_session_contents",
        lambda _session_code: [
            {"role": "user", "property": {"turn_id": "turn-existing"}},
        ],
    )

    session_code, turn_id = session_manager.prepare_session_turn("thread-1", input_text="")

    assert session_code
    assert turn_id == "turn-existing"
    mock_plugin_rm_client.api.create_chat_session_content.assert_not_called()
