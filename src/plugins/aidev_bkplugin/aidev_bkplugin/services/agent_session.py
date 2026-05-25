# -*- coding: utf-8 -*-
"""Session 管理。

封装 chat session 的取/建、内容持久化；底层 client 通过
``resource_manager().get_client()`` 获取（应用态 + ``X-BKAIDEV-USER`` header），
不直接依赖 ``aidev_agent.api.bk_aidev.BKAidevApi``。
"""

from __future__ import annotations

import hashlib
import uuid
from logging import getLogger
from typing import Iterable

from aidev_agent.enums import ChatContentStatus, PromptRole, SessionsStatus
from aidev_agent.packages.resource_manager import resource_manager
from aidev_agent.pydantic_models import ChatPrompt
from bkapi_client_core.exceptions import HTTPResponseError
from django.conf import settings

from ..constants import AGUI_PROTOCOL_VERSION
from ..enums import PluginPollTaskState

logger = getLogger(__name__)


class SessionManager:
    """会话管理：取/建 session、持久化 chat history、保存 AI 回复。

    所有 HTTP 调用经由 ``resource_manager().get_client()``，业务侧 import 上不再耦合
    ``BKAidevApi``；用户名通过 ``X-BKAIDEV-USER`` header 透传给后端识别用户。
    """

    def __init__(self, username: str, agent_code: str | None = None):
        self.username = username
        self.agent_code = agent_code or settings.APP_CODE

    @staticmethod
    def generate_session_code(username: str, agent_code: str, thread_id: str) -> str:
        """``MD5(username:agent_code:thread_id)``，确保 session_code 长度固定且对相同三元组稳定。"""
        raw_string = f"{username}:{agent_code}:{thread_id}"
        return hashlib.md5(raw_string.encode()).hexdigest()

    def _client(self):
        return resource_manager().get_client()

    def _user_headers(self) -> dict:
        return {"X-BKAIDEV-USER": self.username}

    def get_or_create_by_thread_id(self, thread_id: str) -> str:
        """根据 ``thread_id`` 取回 session_code；不存在则新建（``protocol_version`` 由模块常量决定）。

        404 之外的 HTTPResponseError 直接抛出，由调用方决定降级策略。
        """
        session_code = self.generate_session_code(self.username, self.agent_code, thread_id)
        client = self._client()
        try:
            result = client.api.retrieve_chat_session(
                path_params={"session_code": session_code},
                headers=self._user_headers(),
            )
            if result.get("data"):
                return session_code
        except HTTPResponseError as err:
            logger.warning("Error retrieving chat session: %s", err)
            if err.response_status_code != 404:
                raise
            client.api.create_chat_session(
                json={
                    "session_code": session_code,
                    "session_name": f"Thread-{thread_id[:8]}",
                    "protocol_version": AGUI_PROTOCOL_VERSION,
                },
                headers=self._user_headers(),
            )
            return session_code
        return session_code

    def save_content(
        self,
        session_code: str,
        role: str,
        content: str,
        *,
        extra: dict | None = None,
        status: str = "success",
        turn_id: str = "",
    ) -> dict:
        """保存单条会话内容到 BKAidev；返回后端 ``data`` 字段。"""
        if role == PromptRole.USER.value and not turn_id:
            turn_id = uuid.uuid4().hex
        data: dict = {"session_code": session_code, "role": role, "content": content, "status": status}
        if extra:
            data["extra"] = extra
        if turn_id:
            data["property"] = {"turn_id": turn_id}
        client = self._client()
        result = client.api.create_chat_session_content(json=data, headers=self._user_headers())
        saved = result.get("data", {})
        if turn_id:
            saved.setdefault("property", {}).setdefault("turn_id", turn_id)
        return saved

    def update_session_status(self, session_code: str, status: str) -> None:
        """更新会话状态，用于标准运维插件后台任务兜底结束态。"""
        self._client().api.update_chat_session(
            path_params={"session_code": session_code},
            json={"status": status},
            headers=self._user_headers(),
        )

    def retrieve_session(self, session_code: str) -> dict:
        result = self._client().api.retrieve_chat_session(
            path_params={"session_code": session_code},
            headers=self._user_headers(),
        )
        return result.get("data") or {}

    def save_chat_history(self, session_code: str, chat_history: Iterable[ChatPrompt] | None) -> None:
        """按顺序持久化 ``chat_history``；不带 ``turn_id``（历史回放 / 批量同步场景）。"""
        for prompt in chat_history or []:
            self.save_content(session_code=session_code, role=prompt.role, content=prompt.content)

    def save_ai_response(self, session_code: str, result: dict, *, turn_id: str = "") -> None:
        """从非流式 LLM 响应中提取 ``choices[0].delta.content`` 并写回；空内容跳过。"""
        content = ""
        if "choices" in result and result["choices"]:
            delta = result["choices"][0].get("delta", {})
            content = delta.get("content", "")

        if content:
            self.save_content(session_code=session_code, role=PromptRole.AI.value, content=content, turn_id=turn_id)

    def list_session_contents(self, session_code: str) -> list[dict]:
        result = self._client().api.get_chat_session_contents(
            params={"session_code": session_code},
            headers=self._user_headers(),
        )
        return result.get("data") or []

    def prepare_session_turn(
        self,
        thread_id: str,
        *,
        input_text: str = "",
        turn_id: str = "",
    ) -> tuple[str, str]:
        """取/建 session，必要时落库本轮 user，返回 ``session_code`` 与 ``turn_id``。"""
        session_code = self.get_or_create_by_thread_id(thread_id)
        if input_text:
            saved = self.save_content(
                session_code=session_code,
                role=PromptRole.USER.value,
                content=input_text,
                turn_id=turn_id,
            )
            turn_id = ((saved.get("property") or {}).get("turn_id") if isinstance(saved, dict) else "") or turn_id
        elif not turn_id:
            # 用户消息已由前端/SDK 落库、本轮无 input 时，继承最近一条 user 的 turn_id
            for item in reversed(self.list_session_contents(session_code)):
                if item.get("role") != PromptRole.USER.value:
                    continue
                turn_id = (item.get("property") or {}).get("turn_id") or ""
                if turn_id:
                    break
        return session_code, turn_id or uuid.uuid4().hex

    def save_stream_failure(self, session_code: str, error_message: str, *, turn_id: str = "") -> None:
        self.save_content(
            session_code=session_code,
            role=PromptRole.ASSISTANT.value,
            content=error_message,
            status=ChatContentStatus.ERROR.value,
            turn_id=turn_id,
        )
        self.update_session_status(session_code, SessionsStatus.FAILED.value)

    def poll_task_state(self, session_code: str, *, turn_id: str = "") -> tuple[PluginPollTaskState, str]:
        """查询本轮 Agent 是否结束。返回 (state, detail)。

        插件 2.0 以 ``session.status`` 判断单轮会话状态。
        """
        status = str(self.retrieve_session(session_code).get("status") or "")
        if status in (SessionsStatus.PENDING.value, SessionsStatus.RUNNING.value):
            return PluginPollTaskState.RUNNING, ""
        if status in (SessionsStatus.FAILED.value, SessionsStatus.CANCELLED.value):
            return PluginPollTaskState.FAILED, "Agent 执行失败"
        if status == SessionsStatus.FINISHED.value:
            return PluginPollTaskState.SUCCESS, self._last_assistant_output(
                self.list_session_contents(session_code),
                turn_id=turn_id,
            )
        return PluginPollTaskState.RUNNING, ""

    @staticmethod
    def _last_assistant_output(items: list[dict], *, turn_id: str = "") -> str:
        """取最后一条有内容的 assistant/ai；有 ``turn_id`` 时仅取同轮消息。"""
        assistant_roles = (PromptRole.ASSISTANT.value, PromptRole.AI.value)
        for item in reversed(items):
            if item.get("role") not in assistant_roles:
                continue
            if turn_id and (item.get("property") or {}).get("turn_id") != turn_id:
                continue
            if item.get("status") not in (ChatContentStatus.COMPLETE.value, ChatContentStatus.SUCCESS.value):
                continue
            text = str(item.get("content") or "").strip()
            if text:
                return text
        return ""
