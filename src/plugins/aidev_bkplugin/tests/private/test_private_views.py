# -*- coding: utf-8 -*-
"""被调方 private 接口：主调方调用权限校验单测。"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aidev_bkplugin.private.permissions import AgentCallPermission
from rest_framework.exceptions import PermissionDenied


def _request(app_code="caller_app", username="alice"):
    app = SimpleNamespace(bk_app_code=app_code, verified=True) if app_code is not None else None
    return SimpleNamespace(app=app, user=SimpleNamespace(username=username))


@pytest.mark.parametrize("allowed, expect_ok", [(True, True), (False, False)])
def test_has_permission_follows_platform_allowed(allowed, expect_ok):
    perm = AgentCallPermission()
    with patch("aidev_bkplugin.private.permissions.resource_manager") as mock_rm:
        mock_rm.return_value.check_agent_call_permission.return_value = {"allowed": allowed}
        if expect_ok:
            assert perm.has_permission(_request(), None) is True
        else:
            with pytest.raises(PermissionDenied):
                perm.has_permission(_request(), None)
        mock_rm.return_value.check_agent_call_permission.assert_called_once_with(
            caller_app_code="caller_app", username="alice"
        )


def test_has_permission_denied_without_caller_app_code():
    perm = AgentCallPermission()
    with patch("aidev_bkplugin.private.permissions.resource_manager") as mock_rm:
        with pytest.raises(PermissionDenied):
            perm.has_permission(_request(app_code=None), None)
        mock_rm.return_value.check_agent_call_permission.assert_not_called()
