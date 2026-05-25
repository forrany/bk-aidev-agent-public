# -*- coding: utf-8 -*-
"""标准运维插件 Celery 任务。"""

from __future__ import annotations

import logging

from aidev_agent.enums import AgentType

from .services.agent_bkplugin import BkpluginChat, BkpluginFlow
from .services.agent_session import SessionManager

try:
    from celery import shared_task
except ImportError:  # pragma: no cover - 未安装 celery 的开发环境会进入
    shared_task = None

logger = logging.getLogger(__name__)

# 与模板 app_desc.yml 中 celery worker 的 -Q 保持一致，否则任务会进默认 celery 队列而无人消费
BKPLUGIN_CELERY_QUEUE = "plugin_schedule"


def _run_bkplugin_background_agent(
    session_code: str,
    execute_kwargs: dict,
    username: str | None,
    agent_type_value: str,
    chat_context: list[dict] | None = None,
) -> None:
    """Celery worker 内执行 Chat / Flow Agent。"""
    manager = SessionManager(username or "")
    try:
        agent_type = AgentType(agent_type_value)
    except ValueError:
        agent_type = AgentType.CHAT

    try:
        logger.info(
            "[Bkplugin] celery task start session_code=%s agent_type=%s",
            session_code,
            agent_type.value,
        )
        if agent_type is AgentType.FLOW:
            BkpluginFlow._run_flow(session_code, execute_kwargs, username)
        else:
            BkpluginChat._run_chat(session_code, execute_kwargs, username, chat_context or [])
    except Exception as e:
        logger.exception("[Bkplugin] celery task error session_code=%s: %s", session_code, e)
        manager.save_stream_failure(
            session_code,
            f"Agent 执行异常: {e}",
            turn_id=execute_kwargs.get("turn_id") or "",
        )
    finally:
        logger.info("[Bkplugin] celery task finished session_code=%s", session_code)


if shared_task is not None:

    @shared_task(
        name="aidev_bkplugin.run_background_agent",
        ignore_result=True,
        queue=BKPLUGIN_CELERY_QUEUE,
    )
    def run_bkplugin_background_agent_task(
        session_code: str,
        execute_kwargs: dict,
        username: str | None,
        agent_type_value: str,
        chat_context: list[dict] | None = None,
    ) -> None:
        _run_bkplugin_background_agent(session_code, execute_kwargs, username, agent_type_value, chat_context)

else:
    run_bkplugin_background_agent_task = None
