# -*- coding: utf-8 -*-
"""``user_operation.dispatch`` 行为契约：薄代理 / 透传到平台。

校验点：
- 三种 operation 均按 ``json=data`` + ``headers={X-BKAIDEV-USER}`` 转发到平台
  ``client.api.user_operation``；
- 平台响应 ``{"data": envelope}`` 形态被剥壳，envelope 原样返回前端；
- 平台调用抛异常时，dispatch 统一封装为 ``ClientBlueException``。
"""

import sys
from unittest.mock import MagicMock

import pytest
from blueapps.core.exceptions import ClientBlueException

# 本测试只使用 ``user_operation.dispatch``，与 ORM Model 无关，安全 mock。
if "pkg_resources" not in sys.modules:
    sys.modules["pkg_resources"] = MagicMock()

_bk_plugin_framework = MagicMock()
_bk_plugin_framework.kit.decorators.inject_user_token = lambda func: func
sys.modules.setdefault("bk_plugin_framework", _bk_plugin_framework)
sys.modules.setdefault("bk_plugin_framework.kit", _bk_plugin_framework.kit)
sys.modules.setdefault("bk_plugin_framework.kit.decorators", _bk_plugin_framework.kit.decorators)

_fake_models = MagicMock()
_fake_models.Checkpoint = MagicMock()
_fake_models.Write = MagicMock()
sys.modules.setdefault("aidev_bkplugin.models", _fake_models)

from aidev_bkplugin.services import user_operation as user_operation_mod  # noqa: E402


@pytest.fixture
def mock_client(monkeypatch):
    """拦截 ``AgentHelper.get_client``，返回带 ``api.user_operation`` 的 MagicMock。"""
    client = MagicMock()
    client.api.user_operation.return_value = {
        "data": {
            "ok": True,
            "operation": "stub",
            "session_code": "sc-1",
            "next": {"endpoint": "chat_completion", "payload": {"session_code": "sc-1"}},
            "result": {},
        }
    }
    monkeypatch.setattr(user_operation_mod.AgentHelper, "get_client", staticmethod(lambda: client))
    return user_operation_mod, client


@pytest.mark.parametrize(
    "operation, payload",
    [
        ("flow_node_retry", {"task_id": "t1", "node_id": "n1"}),
        ("flow_node_skip", {"task_id": "t1", "node_id": "n1"}),
        ("approval_cancel", {"interrupt_id": "int-approval-call_x"}),
    ],
)
def test_dispatch_proxies_to_platform_with_user_header(mock_client, operation, payload):
    """三种 operation 均透传 data + X-BKAIDEV-USER 头到平台。"""
    mod, client = mock_client
    data = {"session_code": "sc-1", "operation": operation, "payload": payload}

    envelope = mod.dispatch(operation, "alice", data)

    # 平台被以 json=data, X-BKAIDEV-USER 头调用
    client.api.user_operation.assert_called_once_with(
        json=data, headers={"X-BKAIDEV-USER": "alice"}
    )
    # envelope 剥壳后透传
    assert envelope["ok"] is True
    assert envelope["session_code"] == "sc-1"


def test_dispatch_returns_envelope_when_response_already_unwrapped(mock_client):
    """兼容平台直接返回 envelope 形态（无 ``data`` 包装，用于开发期 / 单测）。"""
    mod, client = mock_client
    client.api.user_operation.return_value = {
        "ok": True,
        "operation": "approval_cancel",
        "session_code": "sc-2",
        "next": {},
        "result": {},
    }

    envelope = mod.dispatch(
        "approval_cancel",
        "bob",
        {"session_code": "sc-2", "operation": "approval_cancel", "payload": {"interrupt_id": "x"}},
    )
    assert envelope["operation"] == "approval_cancel"
    assert envelope["session_code"] == "sc-2"


def test_dispatch_wraps_platform_exception_as_client_blue(mock_client):
    """平台抛异常时统一封装为 ClientBlueException。"""
    mod, client = mock_client
    client.api.user_operation.side_effect = RuntimeError("platform boom")

    with pytest.raises(ClientBlueException):
        mod.dispatch(
            "flow_node_retry",
            "alice",
            {"session_code": "sc-1", "operation": "flow_node_retry", "payload": {"task_id": "t1", "node_id": "n1"}},
        )


def test_dispatch_rejects_non_dict_envelope(mock_client):
    """平台返回非 dict envelope 时显式失败（避免静默错误传给前端）。"""
    mod, client = mock_client
    client.api.user_operation.return_value = {"data": "not-a-dict"}

    with pytest.raises(ClientBlueException):
        mod.dispatch(
            "approval_cancel",
            "alice",
            {"session_code": "sc-1", "operation": "approval_cancel", "payload": {"interrupt_id": "x"}},
        )


def test_dispatch_without_username_omits_header(mock_client):
    """username 为空时不注入 X-BKAIDEV-USER 头，由平台兜底鉴权。"""
    mod, client = mock_client
    data = {
        "session_code": "sc-1",
        "operation": "approval_cancel",
        "payload": {"interrupt_id": "x"},
    }

    mod.dispatch("approval_cancel", "", data)

    client.api.user_operation.assert_called_once_with(json=data, headers={})
