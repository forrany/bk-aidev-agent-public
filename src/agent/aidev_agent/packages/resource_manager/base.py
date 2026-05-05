# -*- coding: utf-8 -*-
"""``BaseResourceManager``：资源管理器通用业务基类。

本基类收敛资源方法的业务装配和构造函数：
- 构造函数接收 app_code / app_secret / access_token / username 等通用参数；
- path_params / params / data 的组织；
- 后端响应 ``data`` 字段抽取；
- tool 结构转换。

Client 的创建、认证信息注入由 ``get_client()`` 负责，子类可覆写。
"""

from __future__ import annotations

import abc
import json
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.tools import StructuredTool

from aidev_agent.config import settings
from aidev_agent.enums import CredentialType
from aidev_agent.packages.langchain_core.tools import Tool, ToolExtra, make_structured_tool
from aidev_agent.pydantic_models import AgentConfig, AgentOptions, IntentRecognition, KnowledgebaseSettings

if TYPE_CHECKING:
    from aidev_agent.api.bk_aidev import Client


class BaseResourceManager(abc.ABC):
    """资源管理器通用业务基类。

    子类需实现 ``get_client()`` 返回已完成认证注入的 ``Client``。

    :param app_code: 应用编码，用于创建 API Client。
    :param app_secret: 应用密钥，用于创建 API Client。
    :param access_token: 用户 access_token，注入到 X-Bkapi-Authorization。
    :param username: 用户名
    """

    def __init__(
        self,
        app_code: str = "",
        app_secret: str = "",
        access_token: str = "",
        username: str = "",
        **kwargs: Any,
    ) -> None:
        self.app_code = app_code or settings.APP_CODE
        self.app_secret = app_secret or settings.SECRET_KEY
        self.access_token = access_token
        self.username = username
        self._extra = kwargs

    def get_client(self, **kwargs: Any) -> Client:
        """获取已完成认证信息注入的 API Client。子类必须实现。"""
        raise NotImplementedError

    # ---------- 资源方法 (7) ----------

    def retrieve_knowledgebase(self, id: int, **kwargs) -> dict:
        client = self.get_client()
        return client.api.appspace_retrieve_knowledgebase(path_params={"id": id}, **kwargs).get("data", {})

    def retrieve_knowledge(self, id: int, **kwargs) -> dict:
        client = self.get_client()
        return client.api.appspace_retrieve_knowledge(path_params={"id": id}, **kwargs).get("data", {})

    def get_chat_session_context(self, session_code: str, **kwargs) -> list[dict]:
        client = self.get_client()
        return client.api.get_chat_session_context(path_params={"session_code": session_code}, **kwargs).get("data", [])

    def retrieve_agent_config(self, agent_code: str, version: Optional[str] = None, **kwargs) -> dict:
        params = kwargs.pop("params", None) or {}
        if version is not None:
            params["version"] = version
        if params:
            kwargs["params"] = params
        client = self.get_client()
        return client.api.retrieve_agent_config(path_params={"agent_code": agent_code}, **kwargs).get("data", {})

    def get_agent_config(self, agent_code: str, version: Optional[str] = None, **kwargs) -> AgentConfig:
        """取回装配后的 ``AgentConfig``。

        - 透传 ``version`` 给 ``retrieve_agent_config``；为 ``None`` 时由后端返回最新版。
        - 取回失败统一抛 ``ValueError``，与历史 ``AgentConfigManager.get_config`` 行为对齐。
        - 装配规则（与历史一致）：
          * ``KnowledgebaseSettings``：未设置 ``is_response_when_no_knowledgebase_match`` 且无 ``rejection_message``
            时回填默认拒答文案；
          * ``prompt_setting`` 中 ``llm_token_limit`` 合并到 ``KnowledgebaseSettings``；
          * ``prompt_setting`` 中 ``tool_output_compress_thrd`` 合并到 ``IntentRecognition``；
          * ``conversation_settings.commands`` → ``command_agent_mapping``。
        """
        try:
            res = self.retrieve_agent_config(agent_code, version=version, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to retrieve agent config: {e}")

        prompt_setting = res.get("prompt_setting", {}) or {}
        role_prompts = prompt_setting.get("content")
        knowledgebase_settings_data = res.get("knowledgebase_settings") or {}
        intent_recognition_data = res.get("intent_recognition") or {}

        if not knowledgebase_settings_data.get(
            "is_response_when_no_knowledgebase_match"
        ) and not knowledgebase_settings_data.get("rejection_message"):
            knowledgebase_settings_data["rejection_message"] = (
                KnowledgebaseSettings().model_validate({}).rejection_message
            )

        if prompt_setting.get("llm_token_limit") is not None:
            knowledgebase_settings_data["llm_token_limit"] = prompt_setting.get("llm_token_limit")
        if prompt_setting.get("tool_output_compress_thrd") is not None:
            intent_recognition_data["tool_output_compress_thrd"] = prompt_setting.get("tool_output_compress_thrd")

        conversation_settings = res.get("conversation_settings", {}) or {}
        return AgentConfig(
            agent_code=agent_code,
            agent_name=res["agent_name"],
            chat_model=prompt_setting.get("llm_code", ""),
            non_thinking_llm=prompt_setting.get("non_thinking_llm") or prompt_setting.get("llm_code", ""),
            role_prompts=role_prompts or None,
            knowledgebase_ids=res["knowledgebase_settings"]["knowledgebases"],
            tool_codes=res["related_tools"],
            opening_mark=conversation_settings.get("opening_remark") or None,
            mcp_server_config=res.get("mcp_server_config", {}).get("mcpServers", {}),
            related_skills=res.get("related_skills"),
            agent_options=AgentOptions(
                intent_recognition_options=IntentRecognition.model_validate(intent_recognition_data),
                knowledge_query_options=KnowledgebaseSettings.model_validate(knowledgebase_settings_data),
            ),
            command_agent_mapping={
                each["id"]: each["agent_code"] for each in conversation_settings.get("commands", [])
            },
            temperature=prompt_setting.get("temperature"),
            max_tokens=prompt_setting.get("max_tokens"),
        )

    def retrieve_skill(self, skill_id: str, version: str, **kwargs) -> dict:
        params = kwargs.pop("params", {})
        params["version"] = version
        client = self.get_client()
        return client.api.retrieve_resource_v1_skill(path_params={"skill_id": skill_id}, params=params, **kwargs).get(
            "data", {}
        )

    def construct_tool(self, tool_code: str, **kwargs) -> StructuredTool:
        operation_name = "retrieve_tool" if kwargs.pop("appspace", True) else "appspace_retrieve_tool"
        client = self.get_client()
        operation = getattr(client.api, operation_name)
        result = operation(path_params={"tool_code": tool_code}, **kwargs)
        result["data"]["tool_cn_name"] = result["data"]["tool_name"]
        if result["data"].get("credential_type", "") != CredentialType.NULL.value:
            tool = Tool.model_validate(result["data"])
            tool.extra = ToolExtra(
                header={
                    "X-Bkapi-Authorization": json.dumps(
                        {"bk_app_code": self.app_code, "bk_app_secret": self.app_secret}
                    )
                }
            )
            return make_structured_tool(tool)
        return make_structured_tool(Tool.model_validate(result["data"]))

    def knowledge_query(self, data: dict[str, Any]) -> dict:
        client = self.get_client()
        return client.api.create_knowledgebase_query(data=data).get("data", {})

    # ---------- Flow Agent 方法 (8) ----------

    def start_flow_agent(self, data: dict, **kwargs) -> dict:
        client = self.get_client()
        return client.api.flow_agent_start(data=data, **kwargs).get("data", {})

    def get_flow_agent_task_info(self, task_id: str, **kwargs) -> dict:
        client = self.get_client()
        return client.api.flow_agent_task_info(path_params={"task_id": task_id}, **kwargs).get("data", {})

    def retry_flow_agent_node(self, session_code: str, node_id: str, **kwargs) -> dict:
        client = self.get_client()
        return client.api.flow_agent_retry_node(
            path_params={"session_code": session_code, "node_id": node_id}, **kwargs
        ).get("data", {})

    def skip_flow_agent_node(self, session_code: str, node_id: str, **kwargs) -> dict:
        client = self.get_client()
        return client.api.flow_agent_skip_node(
            path_params={"session_code": session_code, "node_id": node_id}, **kwargs
        ).get("data", {})

    def stop_flow_agent_task(self, session_code: str, **kwargs) -> dict:
        client = self.get_client()
        return client.api.flow_agent_task_stop(data={"session_code": session_code}, **kwargs).get("data", {})

    def pause_flow_agent_task(self, session_code: str, **kwargs) -> dict:
        client = self.get_client()
        return client.api.flow_agent_task_pause(data={"session_code": session_code}, **kwargs).get("data", {})

    def resume_flow_agent_task(self, session_code: str, **kwargs) -> dict:
        client = self.get_client()
        return client.api.flow_agent_task_resume(data={"session_code": session_code}, **kwargs).get("data", {})

    def get_flow_agent_task_node_info(self, task_id: str, node_id: str, **kwargs) -> dict:
        client = self.get_client()
        return client.api.flow_agent_task_node_info(path_params={"task_id": task_id, "node_id": node_id}, **kwargs).get(
            "data", {}
        )
