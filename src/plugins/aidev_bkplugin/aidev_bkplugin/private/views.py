# -*- coding: utf-8 -*-

"""被调方智能体 private 接口。

此入口供其它智能体（主调方）通过 apigw 调用本智能体（被调方）。与 openapi 入口的区别在于
权限类 ``AgentCallPermission``：以被调方自身 app_code 调用平台
``/openapi/aidev/resource/v1/agent/call_permission/check/`` 校验主调方是否有调用权限，
校验通过后直接透传到复用 openapi 实现的子类。
"""

from logging import getLogger

from aidev_agent.packages.resource_manager import resource_manager
from rest_framework.response import Response

from aidev_bkplugin.openapi.views import OpenapiChatCompletionViewSet, OpenapiPluginViewSet
from aidev_bkplugin.packages.apigw.permissions import ApigwPermission
from aidev_bkplugin.private.permissions import AgentCallPermission
from aidev_bkplugin.views.base import PluginViewSet

logger = getLogger(__name__)


class PrivatePluginViewSet(OpenapiPluginViewSet):
    """private 入口基类：apigw 来源 + 主调方调用权限校验，用户信息沿用 X-BKAIDEV-USER。"""

    # 与 openapi 的区别：用 AgentCallPermission 替换 AgentPluginPermission
    permission_classes = [ApigwPermission, AgentCallPermission]


class PrivateChatCompletionViewSet(PrivatePluginViewSet, OpenapiChatCompletionViewSet):
    """LLM 会话接口：逻辑复用 openapi 实现，仅权限校验为被调方调用鉴权。"""


class PrivatePingViewSet(PluginViewSet):
    """存活/自检接口：不走权限类，直接调用平台校验主调方调用权限并返回结果。"""

    permission_classes = []

    def list(self, request, *args, **kwargs):
        caller_app_code = AgentCallPermission._get_caller_app_code(request)
        username = getattr(request.user, "username", "")
        result = resource_manager().check_agent_call_permission(
            caller_app_code=caller_app_code,
            username=username,
        )
        return Response(data=result)
