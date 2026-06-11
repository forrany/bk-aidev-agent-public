# -*- coding: utf-8 -*-

"""被调方 private 接口路由。

1. 上层入口：bk_plugin/private/urls.py（由 bk_plugin/patch/urls.py 挂载到 /bk_plugin/private/）
2. 主调方为其它智能体，调用权限由 AgentCallPermission 调平台接口校验
3. 使用用户由主调方通过 Header X-BKAIDEV-USER 传入
"""

from django.urls import include, re_path
from rest_framework.routers import DefaultRouter

from aidev_bkplugin.private.views import PrivateChatCompletionViewSet, PrivatePingViewSet

_router = DefaultRouter()
_router.register("chat_completion", PrivateChatCompletionViewSet, "private_chat_completion")
_router.register("ping", PrivatePingViewSet, "private_ping")


urlpatterns = [
    re_path("agent/", include(_router.urls)),
]
