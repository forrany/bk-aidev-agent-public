# -*- coding: utf-8 -*-
"""统一用户操作入口。

POST ``/user_operation/`` —— 一个 endpoint cover 三种动作：

- ``flow_node_retry`` / ``flow_node_skip``：触发 bkflow 节点重试/跳过；
- ``approval_cancel``：用户主动取消工具审批，写 ``approve_result="cancelled"``。

实现策略：plugin 端仅做入参校验 + 鉴权语境解析，业务逻辑下沉到平台
``/openapi/aidev/resource/v1/chat/user_operation/``，由 :func:`dispatch` 透传调用。
所有 DB 写动作（``approve_result`` 落库、``flow_info.resume_pending`` 置位、
bkflow 节点状态切换）均由平台完成，plugin 不持有 schema 知识。

成功统一返回 JSON 信封，前端根据 ``next.payload`` 再调 ``chat_completion`` 续流。
失败抛 ``ClientBlueException``，**不返回 SSE**。
"""

from logging import getLogger

from rest_framework.views import Response

from aidev_bkplugin.serializers.user_operation import UserOperationSerializer
from aidev_bkplugin.services.user_operation import dispatch
from aidev_bkplugin.views.base import PluginViewSet

logger = getLogger(__name__)


class UserOperationViewSet(PluginViewSet):
    """统一用户操作 ViewSet，仅 ``POST`` 一个动作。"""

    def create(self, request):
        username = self.get_username()

        serializer = UserOperationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        operation = data["operation"]
        session_code = data["session_code"]
        logger.info(
            "[UserOperation] dispatch start: username=%s, operation=%s, session_code=%s",
            username,
            operation,
            session_code,
        )

        envelope = dispatch(operation, username, data)
        return Response(data=envelope)
