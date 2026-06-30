# -*- coding: utf-8 -*-
"""UserOperationViewSet 业务编排层（薄代理 / 透传到平台）。

把 ``operation`` 透传给平台 ``/openapi/aidev/resource/v1/chat/user_operation/`` 接口，
由平台完成所有 DB 写动作（``approval_cancel`` / ``flow_node_retry`` /
``flow_node_skip``）。
"""

from logging import getLogger

from blueapps.core.exceptions import ClientBlueException

from aidev_bkplugin.services.agent_helpers import AgentHelper

logger = getLogger(__name__)


def dispatch(operation: str, username: str, data: dict) -> dict:
    """透传 ``user_operation`` 请求到平台，返回响应信封 dict。

    Args:
        operation: 操作类型（``approval_cancel`` / ``flow_node_retry`` /
            ``flow_node_skip``），由 serializer 已校验。
        username: 当前用户名（plugin 鉴权语境解析得到，作为 ``X-BKAIDEV-USER``
            头透传给平台；平台据此做对象级鉴权）。
        data: serializer ``validated_data``，原样转发给平台。

    Returns:
        平台 ``user_operation`` 接口返回的响应信封 dict。

    Raises:
        ClientBlueException: 平台返回失败（4xx/5xx）或网络异常。
    """
    client = AgentHelper.get_client()
    headers = {"X-BKAIDEV-USER": username} if username else {}

    logger.info(
        "[UserOperation] proxy to platform: username=%s, operation=%s, session_code=%s",
        username,
        operation,
        data.get("session_code"),
    )

    try:
        result = client.api.user_operation(json=data, headers=headers)
    except Exception as err:
        logger.exception(
            "[UserOperation] proxy to platform failed: operation=%s, session_code=%s, err=%s",
            operation,
            data.get("session_code"),
            err,
        )
        message = getattr(err, "message", None) or str(err) or f"user_operation {operation} failed"
        raise ClientBlueException(message=message)

    # 平台响应统一形态：{"result": true, "data": <envelope>, ...}
    # 兼容直接返回 envelope 的旧形态（开发期 / 单测）
    if isinstance(result, dict) and "data" in result:
        envelope = result.get("data")
    else:
        envelope = result
    if not isinstance(envelope, dict):
        raise ClientBlueException(
            message=f"unexpected user_operation response: {result!r}",
        )
    return envelope
