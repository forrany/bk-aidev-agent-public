# -*- coding: utf-8 -*-
"""``services/agent_*`` 共用 mock：拦截 ``SessionManager._client()`` 返回的 client。

业务侧 ``SessionManager`` 内部通过 ``self._client()`` 取 client（``resource_manager().get_client()``）；
为避免单测真实 HTTP，本 conftest 把 ``SessionManager._client`` 替换为返回 ``MagicMock`` 的方法，
单测内取 ``mock_client = SessionManager(...)._client()`` 即可断言 ``client.api.xxx`` 调用。
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_plugin_rm_client(monkeypatch):
    client = MagicMock()
    from aidev_bkplugin.services.agent_session import SessionManager

    monkeypatch.setattr(SessionManager, "_client", lambda self: client)
    return client
