# -*- coding: utf-8 -*-
"""AG-UI 会话回写器 - API 方式

通过 BKAidev API 将 Agent 事件回写到平台数据库。
适用于插件、第三方应用等需要通过 API 访问平台的场景。
"""

from typing import Any

from aidev_agent.api.bkaidev_client.client import Client
from aidev_agent.services.event_handlers.base import BaseSessionWriter


class AGUISessionWriter(BaseSessionWriter):
    """AG-UI 会话回写器（API 方式）

    通过 BKAidev API Client 将 Agent 事件回写到平台。

    Example:
        ```python
        from aidev_agent.api.bkaidev_client.client import Client
        from aidev_agent.services.event_handlers import AGUISessionWriter

        client = Client(...)
        writer = AGUISessionWriter(
            session_code="xxx",
            client=client,
            username="admin",
            tools=tools,  # 可选，用于获取工具描述信息
        )

        # 作为 event_handler 传入 ChatCompletionAgent
        agent = ChatCompletionAgent(
            ...,
            event_handler=writer,
        )
        ```
    """

    def __init__(
        self,
        session_code: str,
        client: Client,
        username: str = "",
        tools: list | None = None,
    ):
        """初始化 API 回写器

        Args:
            session_code: 会话标识
            client: BKAidev API 客户端
            username: 用户名
            tools: 工具列表，用于获取工具描述信息
        """
        super().__init__(session_code=session_code, username=username, tools=tools)
        self.client = client

    def _do_create_content(self, payload: dict[str, Any], headers: dict[str, str]) -> None:
        """通过 API 创建会话内容"""
        self.client.api.create_chat_session_content(json=payload, headers=headers)
