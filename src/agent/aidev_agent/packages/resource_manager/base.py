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
import asyncio
import json
import os
import time
from copy import deepcopy
from logging import getLogger
from typing import TYPE_CHECKING, Any, List, Optional

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from aidev_agent.api.paas_client import BkPaaSSandboxApi
from aidev_agent.config import settings
from aidev_agent.enums import CredentialType
from aidev_agent.packages.langchain_core.tools import Tool, ToolExtra, make_structured_tool
from aidev_agent.packages.langchain_core.tools.base import (
    MCPExceptionWrapper,
    McpToolFetchFailure,
    McpToolsResult,
    _extract_mcp_tools_error_detail,
)
from aidev_agent.pydantic_models import AgentConfig
from aidev_agent.utils.loop import run_coro_sync

try:
    import bkoauth
except ImportError:
    bkoauth = None

if TYPE_CHECKING:
    from aidev_agent.api.bk_aidev import Client

_logger = getLogger(__name__)


def _get_access_token_by_user(username: str) -> str | None:
    """通过 bkoauth 获取用户 access_token（延迟访问，依赖 Django AppConfig.ready 调用 _init_function）。"""
    fn = getattr(bkoauth, "get_access_token_by_user", None) if bkoauth else None
    if fn is None:
        return None
    try:
        token = fn(username)
        return getattr(token, "access_token", None) if token else None
    except Exception as e:
        _logger.warning(f"Failed to get access_token by username {username}: {e}")
        return None


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

    def get_agent_code(self, **kwargs: Any) -> str:
        """获取resource manager的agent code。子类可覆写按照其他场景获取agent code"""
        return self.app_code

    def resolve_access_token(self, username: str = None) -> str:
        """获取 access_token，优先级：self.access_token > username 参数 > self.username > 空字符串。

        :param username: 用户名，用于 fallback 获取 access_token；未传入时使用 self.username
        :return: access_token 字符串
        """
        access_token = self.access_token or ""
        _username = username or self.username
        if not access_token and _username:
            access_token = _get_access_token_by_user(_username) or ""
            if not access_token:
                _logger.warning(
                    f"[credential] resolve_access_token: empty result, "
                    f"app_code={self.app_code}, username={_username}, rm_type={type(self).__name__}"
                )
        return access_token

    def get_paas_sbx_client(self, executor_info: dict, **kwargs) -> Any:

        app_code = executor_info.get("app_code", "")
        app_secret = executor_info.get("app_secret", "")
        bk_username = executor_info.get("executor", "")
        access_token = executor_info.get("access_token", "")

        if app_code and app_secret:
            client = BkPaaSSandboxApi.get_client(app_code=app_code, app_secret=app_secret)
        else:
            client = BkPaaSSandboxApi.get_client_by_username(bk_username)

        client.update_bkapi_authorization(
            access_token=access_token or None,
            bk_username=bk_username or ""
        )
        return client

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

    def retrieve_chat_session(self, session_code: str, **kwargs) -> dict:
        client = self.get_client()
        return client.api.retrieve_chat_session(path_params={"session_code": session_code}, **kwargs).get("data", {})

    def update_chat_session_sandbox_pv_id(self, session_code: str, sandbox_pv_id: str, **kwargs) -> dict:
        client = self.get_client()
        return client.api.update_chat_session(
            path_params={"session_code": session_code},
            json={"session_property": {"sandbox_pv_id": sandbox_pv_id}},
            **kwargs,
        ).get("data", {})

    def retrieve_agent_config(self, agent_code: str, version: Optional[str] = None, **kwargs) -> dict:
        agent_code = agent_code or self.app_code
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
          * ``prompt_setting`` / ``knowledgebase_settings`` / ``intent_recognition`` → ``model_context_options_data``；
          * ``conversation_settings.commands`` → ``command_agent_mapping``。
        """
        try:
            res = self.retrieve_agent_config(agent_code, version=version, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to retrieve agent config: {e}")

        # 模型上下文相关的配置
        prompt_setting = res.get("prompt_setting", {}) or {}
        role_prompts = prompt_setting.get("content")
        knowledgebase_settings_data = res.get("knowledgebase_settings") or {}
        intent_recognition_data = res.get("intent_recognition") or {}

        # 构建知识库查询相关配置，由于历史原因，有一部分值在平台保存在 intent_recognition_data 中
        # 这一部分现在需要移动到 knowledge_query_options_data 中
        knowledge_query_options_data = dict(knowledgebase_settings_data)
        for key in (
            "with_index_specific_search_init",
            "with_index_specific_search_translation",
            "with_index_specific_search_keywords",
        ):
            if key in intent_recognition_data:
                knowledge_query_options_data[key] = intent_recognition_data[key]

        # 平台字段映射：document_fragment_count > 0 时映射为 knowledge_resource_rough_recall_topk
        if (
            "document_fragment_count" in knowledge_query_options_data
            and knowledge_query_options_data["document_fragment_count"] > 0
        ):
            knowledge_query_options_data["knowledge_resource_rough_recall_topk"] = knowledge_query_options_data.pop(
                "document_fragment_count"
            )

        # 平台可能返回空字符串 rejection_message，pop 掉以使用 Pydantic 默认值
        if knowledge_query_options_data.get("rejection_message") == "":
            knowledge_query_options_data.pop("rejection_message")

        # 构建模型上下文需要的值，主要来源是 prompt_setting
        model_context_options_data = dict(prompt_setting)
        if intent_recognition_data.get("agent_type"):
            model_context_options_data["llm_code_agent_type"] = intent_recognition_data["agent_type"]

        conversation_settings = res.get("conversation_settings", {}) or {}

        # 构造 agent_info：完整的原始配置字典（不对 otel_info 解码，保持原始数据）
        agent_info = dict(res)

        return AgentConfig(
            agent_code=agent_code,
            agent_name=res["agent_name"],
            chat_model=prompt_setting.get("llm_code", ""),
            non_thinking_llm=prompt_setting.get("non_thinking_llm") or prompt_setting.get("llm_code", ""),
            role_prompts=role_prompts,
            knowledgebase_ids=res["knowledgebase_settings"]["knowledgebases"],
            tool_codes=res["related_tools"],
            opening_mark=conversation_settings.get("opening_remark"),
            mcp_server_config=res.get("mcp_server_config", {}).get("mcpServers", {}),
            related_skills=res.get("related_skills"),
            agent_options=None,
            model_context_options_data=model_context_options_data,
            knowledge_query_options_data=knowledge_query_options_data,
            command_agent_mapping={
                each["id"]: each["agent_code"] for each in conversation_settings.get("commands", [])
            },
            temperature=prompt_setting.get("temperature"),
            max_tokens=prompt_setting.get("max_tokens"),
            agent_info=agent_info,
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

    def construct_mcp(
        self,
        mcp_config: dict,
        username: str = None,
        executor_info: dict | None = None,
        **kwargs,
    ) -> Any:
        """按 MCP 配置装配 LangChain ``StructuredTool`` 列表。

        使用 ``langchain_mcp_adapters`` 连接 MCP 服务器并获取工具列表。
        支持凭证处理、selected_tools 过滤、异常处理等功能。

        :param mcp_config: MCP 客户端配置字典，格式为 ``{"server_name": {"url": ..., "transport": ...}}``
        :param username: 用户名，用于 BLUEAPPS 认证
        :param executor_info: 执行用户信息（含 app_code/app_secret/access_token），
            优先用于 MCP 凭证注入，与 skill sandbox 保持一致
        :return: McpToolsResult 对象，包含 tools 和 fetch_failures
        """
        new_server_config = deepcopy(mcp_config)

        # 提取每个 MCP Server 的 selected_tools 配置
        selected_tools_map: dict[str, list[str]] = {}
        for server_name, _server_config in new_server_config.items():
            selected_tools = _server_config.pop("selected_tools", None)
            if selected_tools:
                selected_tools_map[server_name] = selected_tools

        # 处理凭证：优先使用 executor_info 中的凭证（与 skill sandbox 一致），回退到 resource_manager
        _blueapps_servers = []
        _non_blueapps_servers = []
        for _server_config in new_server_config.values():
            if "mcp_type" in _server_config:
                _server_config.pop("mcp_type")
            if _server_config.pop("credential_type", "") == CredentialType.BLUEAPPS.value:
                _blueapps_servers.append(_server_config)
                # 凭证来源：executor_info > resource_manager
                app_code = (executor_info or {}).get("app_code") or self.app_code
                app_secret = (executor_info or {}).get("app_secret") or self.app_secret
                access_token = (executor_info or {}).get("access_token") or self.resolve_access_token(username)
                # 根据是否拿到 access_token 决定认证方式
                if access_token:
                    auth_info = {"access_token": access_token}
                elif username:
                    auth_info = {
                        "bk_app_code": app_code,
                        "bk_app_secret": app_secret,
                        "bk_username": username,
                    }
                else:
                    auth_info = {"bk_app_code": app_code, "bk_app_secret": app_secret}
                _server_config["headers"] = {"X-Bkapi-Authorization": json.dumps(auth_info)}
                _server_config["headers"]["X-Bkapi-Timeout"] = settings.BK_APIGW_MCP_TIMEOUT
            else:
                _non_blueapps_servers.append(_server_config)

        _logger.info(
            f"[credential] construct_mcp: "
            f"app_code={self.app_code}, rm_type={type(self).__name__}, "
            f"blueapps={len(_blueapps_servers)}, non_blueapps={len(_non_blueapps_servers)}, "
            f"from_executor_info={bool(executor_info and executor_info.get('app_code'))}, "
            f"username={username or ''}"
        )

        # 重试2次；返回 (tools, failure | None)，失败时返回 McpToolFetchFailure
        total_servers = len(new_server_config)
        _logger.info(f"[MCP] start loading {total_servers} server(s): {list(new_server_config.keys())}")

        async def _load_tool(server_name, selected_tools_map, index) -> tuple[list[StructuredTool], Any | None]:
            _start = time.monotonic()
            for _i in range(2):
                client = MultiServerMCPClient(new_server_config)
                try:
                    tools: list[StructuredTool] = await client.get_tools(server_name=server_name)
                    total_count = len(tools)
                    if selected_tools_map.get(server_name):
                        tools = [each for each in tools if each.name in selected_tools_map[server_name]]
                    tool_names = [t.name for t in tools]
                    _logger.info(
                        f"[MCP] {index}/{total_servers} server={server_name}: "
                        f"fetched={total_count}, after_filter={len(tools)}, "
                        f"tools={tool_names}, "
                        f"cost={time.monotonic() - _start:.2f}s"
                    )
                    for each in tools:
                        each.coroutine = MCPExceptionWrapper(each.coroutine)
                        if not each.metadata:
                            each.metadata = {}
                        each.metadata["mcp_name"] = server_name
                    return (tools, None)
                except Exception as err:
                    error_detail = _extract_mcp_tools_error_detail(err)
                    error_msg = f"获取MCP工具列表失败: {error_detail}"
                    if _i == 0:
                        continue
                    _logger.warning(
                        f"[MCP] {index}/{total_servers} server={server_name}: failed, "
                        f"cost={time.monotonic() - _start:.2f}s, error={error_msg}",
                        exc_info=err,
                    )
                    return (
                        [],
                        McpToolFetchFailure(
                            server_name=server_name,
                            message=error_msg,
                            error_type=type(err).__name__,
                        ),
                    )

        coros = [_load_tool(server_name, selected_tools_map, i + 1) for i, server_name in enumerate(new_server_config)]

        async def _load_all_tools():
            return await asyncio.gather(*coros)

        coro_results = run_coro_sync(_load_all_tools())
        tools_list: List[StructuredTool] = []
        failures: List[McpToolFetchFailure] = []
        for tlist, fail in coro_results:
            tools_list.extend(tlist)
            if fail is not None:
                failures.append(fail)
        return McpToolsResult(tools=tools_list, fetch_failures=failures)

    def build_skill_env(self, skill_config: dict, username: str = None) -> dict:
        """根据 skill 配置生成沙箱环境变量。

        逻辑与 ``skill_middleware._extract_paas_params`` 中 env_vars 处理保持一致：
        1. 从 ``metadata.bkai_paas_sandbox.envs`` 提取环境变量
        2. 特殊规则：值为 ``None`` 时从环境变量获取
        3. 赋值 ``ACCESS_TOKEN``（从 self.access_token 或 username 获取）

        :param skill_config: skill 配置字典，包含 metadata 等字段
        :param username: 用户名，用于 fallback 获取 access_token
        :return: 环境变量字典
        """
        env_vars = {}

        if skill_config:
            paas_sandbox = skill_config.get("metadata", {}).get("bkai_paas_sandbox", {})
            env_vars = paas_sandbox.get("envs", {})

        # 特殊规则：如果值是 None，则从环境变量中获取
        for key, value in env_vars.items():
            if value is None:
                env_vars[key] = os.getenv(key, "")

        # 赋值 ACCESS_TOKEN：优先级 self.access_token > username > 环境变量
        access_token = self.resolve_access_token(username)

        env_vars["ACCESS_TOKEN"] = access_token or os.getenv("SANDBOX_BP_ACCESS_TOKEN", "")

        return env_vars

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
