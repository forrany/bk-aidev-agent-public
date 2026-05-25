# -*- coding: utf-8 -*-
"""标准运维插件 Agent 编排。

公开用法（推荐）::

    runner = BkpluginAgentRunner.create(...)
    result = runner.execute()

类职责一览：

| 类型 | 职责 |
|------|------|
| ``BkpluginAgentRunner`` | 抽象基类：准备 session/turn、投递 Celery、定义 ``execute()`` |
| ``BkpluginChat`` | Chat 智能体：拼上下文、调 CommonAgent |
| ``BkpluginFlow`` | Flow 智能体：拼 flow_start_params、调流程 Agent |
| ``BkpluginExecuteResult`` | 同步/异步执行结果（``storage`` 供 2.0 POLL） |

``create()`` 按主站 Agent 配置在 Chat / Flow 执行器之间分流；业务代码勿直接 ``BkpluginChat(...)``。
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from aidev_agent.config import settings as agent_settings
from aidev_agent.enums import AgentBuildType, AgentType, PromptRole
from aidev_agent.pydantic_models import ExecuteKwargs
from aidev_agent.services.agent import AgentInstanceFactory
from aidev_agent.services.common_agent import common_agent_factory
from aidev_agent.services.event_handlers.agui_writer import AGUISessionWriter
from aidev_agent.utils.local import request_local
from django.contrib.auth import get_user_model

from .agent_config import AgentConfigFetcher
from .agent_execution import AgentExecutor, build_execute_kwargs
from .agent_helpers import AgentHelper
from .agent_session import SessionManager
from ..enums import PluginPollTaskState
from ..views.base import PluginResourceManager

logger = logging.getLogger(__name__)


@dataclass
class BkpluginExecuteResult:
    """标准运维插件执行结果。

    - 同步：``result`` 为模型/Flow 输出；
    - 异步：``storage`` 供 POLL 阶段写入 ``context.storage``。
    """

    session_code: str
    result: Any = None
    storage: dict | None = None

    @property
    def is_async(self) -> bool:
        return self.storage is not None

    @classmethod
    def async_pending(cls, session_code: str, storage: dict) -> BkpluginExecuteResult:
        return cls(session_code=session_code, storage=storage)

    @classmethod
    def sync_done(cls, session_code: str, result: Any) -> BkpluginExecuteResult:
        return cls(session_code=session_code, result=result)


def resolve_executor_username(executor: str | None) -> str | None:
    """从插件 ``context.data.executor`` 解析用户名。"""
    if not executor:
        return None
    user = get_user_model().objects.filter(username=executor).first()
    if not user:
        return None
    try:
        req = getattr(request_local, "request", None)
        if req is not None:
            req.user = user
    except Exception:
        logger.debug("skip request_local.user: no request context", exc_info=True)
    return user.username


def normalize_execute_kwargs(execute_kwargs: dict | None, *, session_code: str | None = None) -> dict:
    """归一化插件 execute_kwargs，补入 session_code 并修正字段类型。"""
    data = dict(execute_kwargs or {})
    if session_code:
        data["session_code"] = session_code
    if data.get("caller_bk_biz_id"):
        data["caller_bk_biz_id"] = int(data["caller_bk_biz_id"])
    return data


def build_chat_agent_for_session(
    *,
    session_code: str,
    chat_context: list[dict],
    username: str | None,
    version: str | None = None,
    turn_id: str = "",
):
    """构建带 AG-UI 回写器的 Chat Agent 实例（供 execute 与 Celery 共用）。"""
    user = username or ""
    event_handler = AGUISessionWriter(
        session_code=session_code,
        client=AgentHelper.get_client(),
        username=user,
        turn_id=turn_id,
    )
    return AgentInstanceFactory.build_agent(
        build_type=AgentBuildType.DIRECT,
        session_code=session_code,
        session_context_data=chat_context,
        agent_cls=common_agent_factory.get(),
        checkpointer=AgentHelper.get_checkpointer(),
        event_handler=event_handler,
        username=user,
        version=version,
    )


class BkpluginAgentRunner(ABC):
    """单次插件调用的 Agent 编排基类；子类实现 Chat / Flow 的 ``execute()``。"""

    agent_type: ClassVar[AgentType]

    def __init__(
        self,
        *,
        execute_kwargs: dict,
        input_text: str | None = None,
        username: str | None = None,
        plugin_context: list | None = None,
        stream: bool = False,
        parsed_ek: ExecuteKwargs | None = None,
    ):
        self.execute_kwargs = execute_kwargs
        self.input_text = input_text
        self.username = username
        self.plugin_context = plugin_context or []
        self.stream = stream
        self._parsed_ek = parsed_ek or build_execute_kwargs(execute_kwargs, username)

    @classmethod
    def create(
        cls,
        chat_history: list[dict],
        execute_kwargs: dict,
        input_text: str | None = None,
        username: str | None = None,
        plugin_context: list | None = None,
        stream: bool = False,
    ) -> BkpluginAgentRunner:
        """按主站 Agent 配置构造 Chat 或 Flow 执行器。"""
        parsed_ek = build_execute_kwargs(execute_kwargs, username)
        agent_config = AgentConfigFetcher.get_info(
            username=username or "",
            version=parsed_ek.version,
        )
        configured = agent_config.get("agent_type")
        try:
            resolved = AgentType(configured) if configured else AgentType.CHAT
        except ValueError:
            resolved = AgentType.CHAT
        runner_kwargs = {
            "execute_kwargs": execute_kwargs,
            "input_text": input_text,
            "username": username,
            "plugin_context": plugin_context,
            "stream": stream,
            "parsed_ek": parsed_ek,
        }
        if resolved is AgentType.FLOW:
            return BkpluginFlow(**runner_kwargs)
        return BkpluginChat(chat_history=chat_history, **runner_kwargs)

    @abstractmethod
    def execute(self) -> BkpluginExecuteResult:
        """执行本轮 Agent（同步完成或投递 Celery 后返回 POLL 上下文）。"""

    def _prepare_execution_context(self) -> tuple[str, str, ExecuteKwargs, SessionManager]:
        """编排层：准备 SessionManager 与会话轮次上下文（复用 create 时解析的 execute_kwargs）。"""
        ek = self._parsed_ek
        manager = SessionManager(self.username or "")
        thread_id = ek.session_code or str(uuid.uuid4())
        session_code, turn_id = manager.prepare_session_turn(
            thread_id,
            input_text=self.input_text or "",
            turn_id=ek.turn_id,
        )
        ek.session_code = session_code
        ek.turn_id = turn_id
        return session_code, turn_id, ek, manager

    def _merge_execute_payload(self, ek: ExecuteKwargs, turn_id: str, **extra: Any) -> dict:
        """合并插件原始参数与本轮 execute 字段，供 Celery 或 Flow 启动使用。"""
        return {**self.execute_kwargs, **ek.model_dump(exclude_none=True), "turn_id": turn_id, **extra}

    def _dispatch_async(
        self,
        session_code: str,
        turn_id: str,
        execute_data: dict,
        *,
        chat_context: list[dict] | None = None,
    ) -> BkpluginExecuteResult:
        """2.0：投递 Celery 并返回 POLL storage。"""
        execute_data["stream"] = True
        self.start_background_task(
            session_code,
            execute_data,
            self.agent_type,
            chat_context=chat_context,
        )
        return BkpluginExecuteResult.async_pending(
            session_code,
            self._build_storage(session_code, turn_id, self.agent_type),
        )

    def _build_storage(self, session_code: str, turn_id: str, agent_type: AgentType) -> dict:
        return {
            "session_code": session_code,
            "turn_id": turn_id,
            "plugin_username": self.username or "",
            "agent_type": agent_type.value,
        }

    def start_background_task(
        self,
        session_code: str,
        execute_kwargs: dict,
        agent_type: AgentType,
        *,
        chat_context: list[dict] | None = None,
    ) -> None:
        """投递 Celery 后台任务；POLL 阶段只查状态，不再负责启动。"""
        from ..tasks import run_bkplugin_background_agent_task

        if run_bkplugin_background_agent_task is None:
            raise RuntimeError("Celery is required to run bkplugin async agent")
        run_bkplugin_background_agent_task.delay(
            session_code=session_code,
            execute_kwargs=execute_kwargs,
            username=self.username,
            agent_type_value=agent_type.value,
            chat_context=chat_context or [],
        )


class BkpluginChat(BkpluginAgentRunner):
    agent_type = AgentType.CHAT

    def __init__(self, *, chat_history: list[dict], **kwargs):
        super().__init__(**kwargs)
        self.chat_history = chat_history

    def execute(self) -> BkpluginExecuteResult:
        session_code, turn_id, ek, manager = self._prepare_execution_context()
        chat_context = self._build_chat_context_data()
        if self.stream:
            return self._dispatch_async(
                session_code,
                turn_id,
                self._merge_execute_payload(ek, turn_id),
                chat_context=chat_context,
            )

        ek.stream = False
        agent_instance = build_chat_agent_for_session(
            session_code=session_code,
            chat_context=chat_context,
            username=self.username,
            version=ek.version,
            turn_id=turn_id,
        )
        result = AgentExecutor(manager).execute_with_save(agent_instance, ek, session_code, turn_id=turn_id)
        return BkpluginExecuteResult.sync_done(session_code, result)

    def _build_chat_context_data(self) -> list[dict]:
        context_data = [
            {"role": each["role"], "content": each["content"]}
            for each in (self.chat_history or [])
            if each.get("content")
        ]
        if self.input_text:
            last = context_data[-1] if context_data else None
            if not last or last.get("role") != PromptRole.USER.value or last.get("content") != self.input_text:
                context_data.append({"role": PromptRole.USER.value, "content": self.input_text})
        return context_data

    @staticmethod
    def _run_chat(
        session_code: str,
        execute_kwargs: dict,
        username: str | None,
        chat_context: list[dict],
    ) -> None:
        turn_id = execute_kwargs.get("turn_id") or ""
        manager = SessionManager(username or "")
        ek = build_execute_kwargs(execute_kwargs, username)
        agent = build_chat_agent_for_session(
            session_code=session_code,
            chat_context=chat_context,
            username=username,
            version=ek.version,
            turn_id=turn_id,
        )
        AgentExecutor.run_agent_to_completion(agent, ek, session_code, manager, turn_id=turn_id)


class BkpluginFlow(BkpluginAgentRunner):
    agent_type = AgentType.FLOW

    def execute(self) -> BkpluginExecuteResult:
        session_code, turn_id, ek, _ = self._prepare_execution_context()
        execute_data = self._merge_execute_payload(
            ek,
            turn_id,
            agent_type=self.agent_type.value,
        )
        execute_data["flow_start_params"] = self._build_flow_start_params(session_code)
        if self.stream:
            return self._dispatch_async(session_code, turn_id, execute_data)

        execute_data["stream"] = False
        self._run_flow(session_code, execute_data, self.username)
        state, detail = SessionManager(self.username or "").poll_task_state(session_code, turn_id=turn_id)
        if state == PluginPollTaskState.FAILED:
            raise ValueError(detail or "Agent 执行失败")
        return BkpluginExecuteResult.sync_done(session_code, detail)

    def _build_flow_start_params(self, session_code: str) -> dict:
        params: dict = {"session_code": session_code}
        if self.plugin_context:
            params["context"] = self.plugin_context
        flow_exec = {"executor": self._parsed_ek.executor}
        if self.execute_kwargs.get("timeout") is not None:
            flow_exec["timeout"] = self.execute_kwargs["timeout"]
        params["execute_kwargs"] = flow_exec
        return params

    @staticmethod
    def _run_flow(session_code: str, execute_kwargs: dict, username: str | None) -> None:
        user = username or ""
        params = dict(execute_kwargs.get("flow_start_params") or {})
        params["session_code"] = session_code
        turn_id = execute_kwargs.get("turn_id") or ""
        handler = AGUISessionWriter(
            session_code=session_code,
            client=AgentHelper.get_client(),
            username=user,
            turn_id=turn_id,
        )
        agent = AgentInstanceFactory.build_agent(
            agent_type=AgentType.FLOW,
            build_type=AgentBuildType.DIRECT,
            session_code=session_code,
            session_context_data=[],
            event_handler=handler,
            username=user,
            flow_resource_manager=PluginResourceManager(username=user),
            task_id=params.get("task_id") or execute_kwargs.get("task_id"),
            flow_start_params=params,
            poll_interval=execute_kwargs.get("poll_interval") or agent_settings.FLOW_AGENT_POLL_INTERVAL,
            poll_timeout=execute_kwargs.get("poll_timeout") or agent_settings.FLOW_AGENT_POLL_TIMEOUT,
        )
        # Flow 始终走 AG-UI 事件流；构造带 stream=True 的 ek 以触发 writer 会话状态收尾
        ek = build_execute_kwargs({**execute_kwargs, "stream": True}, username)
        manager = SessionManager(user)
        AgentExecutor.run_agent_to_completion(agent, ek, session_code, manager, turn_id=turn_id)


def record_plugin_poll_failure(storage: dict, error_message: str) -> None:
    """POLL 失败时与后台异常一致：写 error content 并标记 session failed。"""
    session_code = storage.get("session_code") or ""
    if not session_code:
        return
    SessionManager(storage.get("plugin_username") or "").save_stream_failure(
        session_code,
        error_message,
        turn_id=storage.get("turn_id") or "",
    )


def poll_bkplugin_agent(storage: dict) -> tuple[PluginPollTaskState, str, AgentType]:
    """异步轮询只查状态；后台任务在 ``execute(stream=True)`` 时已启动。"""
    session_code = storage.get("session_code") or ""
    if not session_code:
        raise ValueError("poll 缺少 session_code")

    username = storage.get("plugin_username") or None
    try:
        agent_type = AgentType(storage.get("agent_type") or AgentType.CHAT.value)
    except ValueError:
        agent_type = AgentType.CHAT

    state, detail = SessionManager(username or "").poll_task_state(
        session_code,
        turn_id=storage.get("turn_id") or "",
    )
    return state, detail, agent_type
