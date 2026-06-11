"""``BaseResourceManager.check_agent_call_permission`` contract tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from aidev_agent.packages.resource_manager.base import BaseResourceManager


class _StubResourceManager(BaseResourceManager):
    """Resource manager with a mocked platform API client."""

    def __init__(self):
        super().__init__(app_code="callee", app_secret="y")
        self.client = MagicMock()
        self.client.api.check_agent_call_permission.return_value = {"data": {"allowed": True}}

    def get_client(self, **kwargs):
        return self.client


@pytest.mark.parametrize(
    "username, expected_kwargs",
    [
        ("alice", {"headers": {"X-BKAIDEV-USER": "alice"}}),
        (None, {}),
    ],
)
def test_check_agent_call_permission_passes_caller_and_user_header(username, expected_kwargs):
    rm = _StubResourceManager()

    result = rm.check_agent_call_permission("caller_app", username=username)

    assert result == {"allowed": True}
    rm.client.api.check_agent_call_permission.assert_called_once_with(
        data={"caller_app_code": "caller_app"}, **expected_kwargs
    )
