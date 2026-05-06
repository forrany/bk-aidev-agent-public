# -*- coding: utf-8 -*-
"""``ResourceManagerProtocol``：业务侧资源接口契约。

``resource_manager``：全局 :class:`SimpleFactory` 实例工厂。
- 业务侧通过 ``resource_manager()`` 取当前默认实现的实例（无 key）；
- Plugin / 测试侧用 ``resource_manager.replace_defaults(MyImpl)`` 在进程层级整体替换，
  或 ``resource_manager.register(key, callback)`` 按 key 注册多种实现；
- 默认绑定（``AgentResourceManager``）在包 ``__init__.py`` 完成 wiring，
  避免 ``registry.py`` 反向依赖 ``agent.py``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, runtime_checkable

from langchain_core.tools import StructuredTool
from typing_extensions import Protocol

from aidev_agent.utils.factory import SingletonFactory

if TYPE_CHECKING:
    from aidev_agent.api.bk_aidev import Client
    from aidev_agent.pydantic_models import AgentConfig


@runtime_checkable
class ResourceManagerProtocol(Protocol):
    """业务侧资源管理协议。

    所有方法定义业务契约（含返回结构语义），由 ``AgentResourceManager`` 提供默认实现。
    Plugin / 测试侧若要替换或 Mock，按本协议鸭子类型实现即可（不必继承）。
    """

    def get_client(self, **kwargs) -> "Client":
        """获取已完成认证信息注入的 API Client。"""
        ...

    def retrieve_knowledgebase(self, id: int, **kwargs) -> dict:
        """按 ID 取回知识库详情（业务返回结构 = 后端 ``data`` 字段）"""
        ...

    def retrieve_knowledge(self, id: int, **kwargs) -> dict:
        """按 ID 取回知识条目详情（业务返回结构 = 后端 ``data`` 字段）"""
        ...

    def get_chat_session_context(self, session_code: str, **kwargs) -> list[dict]:
        """取回会话上下文消息列表（业务返回结构 = 后端 ``data`` 字段）"""
        ...

    def retrieve_agent_config(self, agent_code: str, version: Optional[str] = None, **kwargs) -> dict:
        """取回 agent 配置原始字典。

        :param version: 可选的 agent 配置版本；为空时由后端返回最新版本。
        """
        ...

    def get_agent_config(self, agent_code: str, version: Optional[str] = None, **kwargs) -> "AgentConfig":
        """取回装配后的 ``AgentConfig``（dict → AgentConfig 装配统一收敛在本协议）。

        默认实现见 ``BaseResourceManager.get_agent_config``；插件 / 测试侧需覆盖时按本协议鸭子类型实现即可。
        """
        ...

    def retrieve_skill(self, skill_id: str, version: str, **kwargs) -> dict:
        """按 skill_id + version 取回技能详情。"""
        ...

    def construct_tool(self, tool_code: str, **kwargs) -> StructuredTool:
        """按 ``tool_code`` 装配 LangChain ``StructuredTool``（含凭证拼装）。"""
        ...

    def resolve_access_token(self, username: str = None) -> str:
        """获取 access_token，优先级：self.access_token > username 参数 > self.username > 空字符串。

        :param username: 用户名，用于 fallback 获取 access_token；未传入时使用 self.username
        :return: access_token 字符串
        """
        ...

    def construct_mcp(self, mcp_config: dict, agent_options: Any = None, username: str = None, **kwargs) -> Any:
        """按 MCP 配置装配 LangChain ``StructuredTool`` 列表。

        使用 ``langchain_mcp_adapters`` 连接 MCP 服务器并获取工具列表。
        """
        ...

    def build_skill_env(self, skill_config: dict, username: str = None) -> dict:
        """按 skill 配置生成沙箱环境变量。

        逻辑与 ``skill_middleware._extract_paas_params`` 中 env_vars 处理保持一致：
        1. 从 ``metadata.bkai_paas_sandbox.envs`` 提取环境变量
        2. 特殊规则：值为 ``None`` 时从环境变量获取
        3. 赋值 ``ACCESS_TOKEN``
        """
        ...

    def knowledge_query(self, data: dict[str, Any]) -> dict:
        """提交知识库查询并返回业务结果（``data`` 字段）。"""
        ...

    def start_flow_agent(self, data: dict, **kwargs) -> dict:
        """启动 flow agent 任务，返回包含 task_id 的字典。"""
        ...

    def get_flow_agent_task_info(self, task_id: str, **kwargs) -> dict:
        """获取 flow agent 任务信息。"""
        ...

    def retry_flow_agent_node(self, session_code: str, node_id: str, **kwargs) -> dict:
        """重试 flow agent 任务节点。"""
        ...

    def skip_flow_agent_node(self, session_code: str, node_id: str, **kwargs) -> dict:
        """跳过 flow agent 任务节点。"""
        ...

    def stop_flow_agent_task(self, session_code: str, **kwargs) -> dict:
        """停止 flow agent 任务。"""
        ...

    def pause_flow_agent_task(self, session_code: str, **kwargs) -> dict:
        """暂停 flow agent 任务。"""
        ...

    def resume_flow_agent_task(self, session_code: str, **kwargs) -> dict:
        """恢复 flow agent 任务。"""
        ...

    def get_flow_agent_task_node_info(self, task_id: str, node_id: str, **kwargs) -> dict:
        """获取 flow agent 任务节点信息。"""
        ...


resource_manager: SingletonFactory[str, ResourceManagerProtocol] = SingletonFactory("resource_manager")
