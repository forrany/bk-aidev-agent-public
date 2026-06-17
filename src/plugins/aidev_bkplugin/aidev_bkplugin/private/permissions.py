# -*- coding: utf-8 -*-

"""被调方 private 接口权限。

``AgentCallPermission`` 以被调方自身 app_code 调用平台
``/openapi/aidev/resource/v1/agent/call_permission/check/`` 校验主调方智能体是否有调用权限。
"""

from logging import getLogger

from aidev_agent.packages.resource_manager import resource_manager
from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import BasePermission

logger = getLogger(__name__)


class AgentCallPermission(BasePermission):
    """被调方校验主调方智能体的调用权限。

    主调方 app_code 取自 apigw（``request.app.bk_app_code``）；使用用户取自 ``request.user.username``
    （由上游 ViewSet 在 ``initial`` 中注入）；被调方以自身 app_code 调用平台 call_permission/check
    接口并据返回的 ``allowed`` 放行或拒绝。
    """

    def has_permission(self, request, view):
        caller_app_code = self._get_caller_app_code(request)
        if not caller_app_code:
            raise PermissionDenied(detail="无法获取主调方 app_code", code="NO_CALLER_APP_CODE")

        username = getattr(request.user, "username", "")
        result = resource_manager().check_agent_call_permission(
            caller_app_code=caller_app_code,
            username=username,
        )
        if not result.get("allowed"):
            raise PermissionDenied(
                detail=f"主调方 {caller_app_code} 没有调用本智能体的权限",
                code="NO_AGENT_CALL_PERMISSION",
            )
        return True

    @staticmethod
    def _get_caller_app_code(request) -> str:
        app = getattr(request, "app", None)
        if app is None:
            return ""
        return getattr(app, "bk_app_code", "") or ""
