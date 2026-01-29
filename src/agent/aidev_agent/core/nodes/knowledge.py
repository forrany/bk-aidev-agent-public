# -*- coding: utf-8 -*-
"""
TencentBlueKing is pleased to support the open source community by making
蓝鲸智云 - AIDev (BlueKing - AIDev) available.
Copyright (C) 2025 THL A29 Limited,
a Tencent company. All rights reserved.
Licensed under the MIT License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the License for the
specific language governing permissions and limitations under the License.
We undertake not to change the open source license (MIT license) applicable
to the current version of the project delivered to anyone in the future.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, List, NotRequired, TypedDict

from langchain_core.callbacks import dispatch_custom_event
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from aidev_agent.core.ag_ui.types import ActivityMessage, CustomMessageType
from aidev_agent.packages.langchain_core.retrievers.bk_retriever import BkRetriever
from aidev_agent.packages.langchain_core.retrievers.kb_rag import KnowledgeRag, KnowledgeRagRetrieveResult
from aidev_agent.services.pydantic_models import AgentOptions

if TYPE_CHECKING:
    from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


class KnowledgeInputState(TypedDict):
    """
    知识库召回的 State 输入字段
    """

    query: NotRequired[str]
    input: NotRequired[str]
    messages: NotRequired[List[BaseMessage]]


class KnowledgeOutputState(KnowledgeRagRetrieveResult):
    pass


def make_knowledge_node(
    llm: BaseChatModel,
    agent_options: AgentOptions,
) -> Callable[[KnowledgeInputState, RunnableConfig, Any], KnowledgeOutputState]:
    """构建知识库检索节点。

    该节点负责:
    - 使用 KnowledgeRag 进行知识库检索
    - 通过 dispatch_custom_event 派发 reference_doc 供前端流式协议使用
    - 将 reference_doc 写入 LangGraph Store 方便后续节点或调用方读取

    Args:
        llm: 语言模型,用于知识库检索中的相关性判断和内容处理
        agent_options: Agent 配置选项,包含知识库配置 (knowledge_bases, knowledge_items)

    Returns:
        可用于 LangGraph node 的 callable,接受 (state, config, *, store) 参数

    Note:
        如果 agent_options 中未配置知识库 (knowledge_bases 和 knowledge_items 均为空),
        则返回一个空实现的节点函数,该函数执行时返回空字典。
    """
    # 如果没有配置知识库，返回空节点实现
    if (
        not agent_options.knowledge_query_options.knowledge_bases
        and not agent_options.knowledge_query_options.knowledge_items
    ):

        def empty_knowledge_node(
            state: Dict[str, Any],
            config: RunnableConfig,
            *,
            store,
        ) -> Dict[str, Any]:
            """空知识库节点实现,当未配置知识库时使用。"""
            return {}

        return empty_knowledge_node

    def knowledge_rag_std_node(
        state: KnowledgeInputState,
        config: RunnableConfig,
        *,
        store,
    ) -> KnowledgeOutputState:
        """知识检索节点实现。

        从 state 中获取用户查询,使用 KnowledgeRag 进行知识库检索,
        并返回包含决策类型、知识内容、引用文档等信息的状态字典。
        """
        t1 = time.time()
        dispatch_custom_event(
            CustomMessageType.KNOWLEDGE_RAG_START.value,
            data={},
            config=config,
        )

        # 获取查询文本,优先使用 query 字段,否则使用 input 字段
        query = state.get("query")
        if query is None:
            query = state.get("input")
        if query is None:
            query = state.get("messages")[-1].content

        # 初始化知识库检索器
        kb_retriever = BkRetriever()
        retriever = KnowledgeRag(llm, kb_retriever)

        # 执行知识库检索,将原始 input 传入便于后续打分等逻辑复用
        ret = retriever.retrieve(query, agent_options, input=query)
        reference_doc = ret.get("reference_doc")

        # 如果有引用文档,进行派发和存储
        message_id = uuid.uuid4().hex
        duration = round(time.time() - t1, 4) * 1000
        if reference_doc:
            # 1. 通过自定义事件向前端推送 reference_doc
            reference_doc = [each["metadata"] for each in reference_doc]
            reference_doc = [
                {"origin_file": each["preview_path"], "url": each["path"], "name": each["display_name"]}
                for each in reference_doc
            ]
            dispatch_custom_event(
                CustomMessageType.KNOWLEDGE_RAG_RESULT.value,
                data={"message_id": message_id, "data": reference_doc, "duration": duration},
                config=config,
            )
            # 2. 将 reference_doc 写入 LangGraph Store,模拟原来的 request_local.current_user_store 行为
            try:
                store.put(("agent", "context"), "reference_doc", reference_doc)
            except Exception:
                logger.warning("写入 reference_doc 到 LangGraph Store 失败", exc_info=True)
            # 3. 在本次节点返回中直接带上 reference_doc,便于非流式调用使用
            ret["reference_doc"] = reference_doc
            # messages
            ret["messages"] = [
                ActivityMessage(
                    activity_type="reference_document",
                    content=reference_doc,
                    id=message_id,
                    additional_kwargs={"duration": duration},
                )
            ]
        else:
            # 如果没有引用文档,派发空消息
            dispatch_custom_event(
                CustomMessageType.KNOWLEDGE_RAG_RESULT.value,
                data={"message_id": message_id, "data": [], "duration": duration},
                config=config,
            )
            ret["messages"] = [
                ActivityMessage(
                    activity_type="reference_document",
                    content=[],
                    id=message_id,
                    additional_kwargs={"duration": duration},
                )
            ]

        dispatch_custom_event(
            CustomMessageType.KNOWLEDGE_RAG_END.value,
            data={},
            config=config,
        )

        return ret

    return knowledge_rag_std_node
