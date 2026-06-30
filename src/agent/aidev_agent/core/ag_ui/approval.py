# -*- coding: utf-8 -*-
"""审批续流（tool approval interrupt resume）相关的 AG-UI 协议适配。

本模块是审批三态字面量以及"终态形态" outcome / result 构造的**单一来源**，
被以下调用方共享，确保 SSE 输出与 DB 落库结构同源：

- ``aidev_agent.core.ag_ui.aidev_agent.AidevAGUIAgent``：续流时的首条
  ``RUN_FINISHED`` 事件构造。
- ``bk-aidev`` 项目侧 ``resource/agent/services/approval.py``（``ToolApprovalResultService``）
  / ``resource/chat/services/user_operation.py``：ITSM 回调与用户主动取消两条
  写库路径的 content 升级写回（``ToolApprovalResultService.write_approval_result``
  内部调用 :meth:`ApprovalOutcomeBuilder.upgrade_content_to_success`）。

模块边界设计：
- ``types.py`` 仅承载纯类型定义（Pydantic / TypedDict / Enum）；本模块承载
  与审批相关的工厂函数与命名空间常量，避免业务逻辑污染类型层。
- 依赖方向严格 ``services -> core``，本模块不反向依赖 services。
"""

from __future__ import annotations

import copy
import json as _json
from typing import Any, Literal


ApproveResultLiteral = Literal["approved", "rejected", "cancelled"]
"""审批三态字面量类型。

与 ``services/agent/approval.py`` 中的同名别名保持一致（后者已 re-export
本模块定义，避免重复定义）。
"""


class ApproveResult:
    """审批结果三态命名空间。

    使用类属性而非 Enum 是为了让字面量值直接是字符串（兼容 DB / JSON 存储），
    同时通过 ``ALL`` 集合提供集中校验入口。
    """

    APPROVED: ApproveResultLiteral = "approved"
    REJECTED: ApproveResultLiteral = "rejected"
    CANCELLED: ApproveResultLiteral = "cancelled"
    ALL: frozenset = frozenset({"approved", "rejected", "cancelled"})


class ApprovalOutcomeBuilder:
    """审批续流终态形态构造器。

    将原始的"中断态" content（``outcome.type == "interrupt"``）升级为"成功态"
    （``outcome.type == "success"`` + 顶层 ``result`` 字段），供：

    - DB ``ChatSessionContent.content`` 字段回写
    - 续流 SSE 首条 ``RUN_FINISHED`` 事件的 ``outcome`` / ``result`` 字段

    目标 content 形态::

        {
            "outcome": {
                "type": "success",
                "interrupts": [
                    {
                        ...,
                        "metadata": {
                            "status": <approve_result>,
                            "ticket": {"status": <approve_result>, ...},
                            ...
                        }
                    }
                ]
            },
            "result": {
                "id": ..., "interruptId": <与 id 同值>,
                "reason": ..., "message": ..., "toolCallId": ...,
                "payload": {"metadata": <interrupts[0].metadata 整体透传>}
            }
        }
    """

    # ------------------------------------------------------------------ #
    # 内部 helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_status_to_interrupt_metadata(
        interrupts: list[dict],
        approve_result: ApproveResultLiteral,
    ) -> None:
        """就地刷写每条 interrupt 的 ``metadata.status`` 与
        ``metadata.ticket.status`` 字段。

        ticket 子结构可能不存在；不存在时仅刷写顶层 status，不强行创建。
        """
        for item in interrupts:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                item["metadata"] = metadata
            metadata["status"] = approve_result
            ticket = metadata.get("ticket")
            if isinstance(ticket, dict):
                ticket["status"] = approve_result

    @staticmethod
    def _build_result_from_first_interrupt(first_interrupt: dict) -> dict:
        """将 ``interrupts[0]`` 扁平化为顶层 ``result`` 对象。

        规则：
        - 顶层保留 ``id`` / ``interruptId`` / ``reason`` / ``message`` /
          ``toolCallId`` 字段；其中 ``interruptId`` 与 ``id`` 同值，供前端按
          中断 id 关联续流结果。
        - 原 ``metadata`` 整体（不裁剪）移入 ``payload.metadata``
        """
        if not isinstance(first_interrupt, dict):
            return {}
        metadata = first_interrupt.get("metadata")
        interrupt_id = first_interrupt.get("id")
        return {
            "id": interrupt_id,
            "interruptId": interrupt_id,
            "reason": first_interrupt.get("reason"),
            "message": first_interrupt.get("message"),
            "toolCallId": first_interrupt.get("toolCallId"),
            "payload": {"metadata": metadata if isinstance(metadata, dict) else {}},
        }

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #

    @classmethod
    def upgrade_content_to_success(
        cls,
        content: Any,
        approve_result: ApproveResultLiteral,
    ) -> dict | None:
        """将审批中断 content 升级为"终态形态"。

        Args:
            content: 原始 content，可以是 dict 或 JSON 字符串。
            approve_result: 三态审批结果。

        Returns:
            改写后的 content dict（深拷贝、不污染入参）。如果 content 结构
            不识别（非 dict / 不含 outcome / outcome.type 既不是 interrupt
            也不是 success / interrupts 为空），返回 ``None``，告知调用方
            无需把 content 字段回写 DB。

        幂等性：对已是 ``success`` 形态的 content 再次传入，会按当前
        ``approve_result`` 重新刷写 ``metadata.status`` / ``ticket.status``
        与顶层 ``result`` 字段。
        """
        if content is None:
            return None
        if isinstance(content, str):
            try:
                content = _json.loads(content)
            except (TypeError, ValueError):
                return None
        if not isinstance(content, dict):
            return None
        outcome = content.get("outcome")
        if not isinstance(outcome, dict):
            return None
        if outcome.get("type") not in ("interrupt", "success"):
            return None
        interrupts = outcome.get("interrupts")
        if not isinstance(interrupts, list) or not interrupts:
            return None

        new_content = copy.deepcopy(content)
        new_outcome = new_content["outcome"]
        new_outcome["type"] = "success"
        new_interrupts = new_outcome["interrupts"]
        cls._apply_status_to_interrupt_metadata(new_interrupts, approve_result)
        new_content["result"] = cls._build_result_from_first_interrupt(new_interrupts[0])
        return new_content

    @classmethod
    def build_run_finished_payload(
        cls,
        interrupts: list[dict],
        approve_result: ApproveResultLiteral,
    ) -> tuple[dict, dict]:
        """构造续流首条 ``RUN_FINISHED`` 事件需要的 ``(outcome, result)`` 字典对。

        与 :meth:`upgrade_content_to_success` 保持完全一致的形态，差异仅在于
        本方法接受 ``interrupts`` 列表（来自
        ``ApprovalStateHandler.fetch_approve_result`` 的返回），返回 tuple
        便于事件构造。

        Args:
            interrupts: 中断列表（dict 形式），允许已包含终态 status 的快照。
            approve_result: 三态字符串，会再次刷写 metadata.status /
                ticket.status，保证一致。

        Returns:
            ``(outcome, result)`` —— ``outcome.type == "success"``；
            ``result`` 为 ``interrupts[0]`` 的扁平化版本。
        """
        safe_interrupts = copy.deepcopy(interrupts) if interrupts else []
        cls._apply_status_to_interrupt_metadata(safe_interrupts, approve_result)
        outcome = {
            "type": "success",
            "interrupts": safe_interrupts,
        }
        result = (
            cls._build_result_from_first_interrupt(safe_interrupts[0])
            if safe_interrupts
            else {}
        )
        return outcome, result


__all__ = [
    "ApproveResultLiteral",
    "ApproveResult",
    "ApprovalOutcomeBuilder",
]
