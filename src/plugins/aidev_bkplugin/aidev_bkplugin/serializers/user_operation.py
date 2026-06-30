# -*- coding: utf-8 -*-
from rest_framework import serializers

from aidev_bkplugin.constants import (
    OPERATION_APPROVAL_CANCEL,
    OPERATION_CHOICES,
    OPERATION_FLOW_NODE_RETRY,
    OPERATION_FLOW_NODE_SKIP,
)


class UserOperationSerializer(serializers.Serializer):
    """
    UserOperationViewSet 入参 serializer。

    统一信封::

        {
          "session_code": "xxx",
          "operation": "flow_node_retry" | "flow_node_skip" | "approval_cancel",
          "payload": { ... },         // 按 operation 不同而要求不同子字段
          "request_id": "..."         // 可选，幂等预留
        }

    operation 取值与 payload 必填项约束（在 ``validate`` 中条件校验）：

    - ``flow_node_retry`` / ``flow_node_skip``：``payload.task_id``（必填）+ ``payload.node_id``（必填）。
    - ``approval_cancel``：``payload.interrupt_id``（必填）。
    """

    session_code = serializers.CharField(required=True)
    operation = serializers.ChoiceField(choices=OPERATION_CHOICES, required=True)
    payload = serializers.DictField(required=False, default=dict)
    request_id = serializers.CharField(required=False, allow_blank=True, default="")

    def validate(self, attrs):
        operation = attrs["operation"]
        payload = attrs.get("payload") or {}

        if operation in (OPERATION_FLOW_NODE_RETRY, OPERATION_FLOW_NODE_SKIP):
            task_id = payload.get("task_id")
            node_id = payload.get("node_id")
            if not task_id:
                raise serializers.ValidationError(
                    {"payload": "task_id is required for flow_node_retry / flow_node_skip"}
                )
            if not node_id:
                raise serializers.ValidationError(
                    {"payload": "node_id is required for flow_node_retry / flow_node_skip"}
                )
        elif operation == OPERATION_APPROVAL_CANCEL:
            interrupt_id = payload.get("interrupt_id")
            if interrupt_id in (None, ""):
                raise serializers.ValidationError(
                    {"payload": "interrupt_id is required for approval_cancel"}
                )

        return attrs
