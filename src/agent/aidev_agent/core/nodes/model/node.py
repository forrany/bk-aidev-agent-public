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
from typing import Annotated, Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph._internal._runnable import RunnableCallable
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from typing_extensions import Required, TypedDict

from aidev_agent.packages.langchain_core.output_parsers import StructuredOutputToToolMessageParser
from aidev_agent.packages.langchain_core.tools.render import render_text_description_and_args

from .context_processor import ContextProcessor
from .pydantic_models import ModelNodeSettings

logger = logging.getLogger(__name__)


def _prepare_model_chain(
    *,
    llm: BaseChatModel,
    context_processor: ContextProcessor,
    use_structured_response: bool,
    enable_parallel_tool_calls: bool,
    state: "ModelState",
    config: RunnableConfig,
    store: BaseStore,
) -> tuple[Any, Dict[str, Any]]:
    """
    准备模型推理所需的 chain 和上下文变量。

    抽取 model_node 和 amodel_node 的公共逻辑，包括：
    - 获取工具列表
    - 获取 prompt 模板
    - 准备上下文变量
    - 根据模式构建 chain

    Args:
        llm: 语言模型
        context_processor: 上下文处理器实例
        use_structured_response: 是否使用结构化输出模式
        enable_parallel_tool_calls: 是否启用并行工具调用
        state: 状态字典
        config: Runnable 配置
        store: LangGraph Store

    Returns:
        tuple: (chain, context_variables)
    """
    # 使用 ContextProcessor 获取工具
    tools = context_processor.get_choice_tools(state, config)
    # 使用 ContextProcessor 选择 prompt 模板
    chat_prompt_template = context_processor.get_chat_prompt_template(state, config, store)
    # 使用 ContextProcessor 准备上下文变量
    context_variables = context_processor.get_chat_prompt_variables(
        chat_prompt_template=chat_prompt_template,
        state=state,
        config=config,
    )

    # 根据模式构建不同的 chain
    if use_structured_response:
        # 创建结构化输出解析器（用于 use_structured_response 模式）
        # 传递 llm 以支持 DeepSeek R1 等模型的特殊处理
        # 注意：不传递 tools，工具存在性校验由 ToolNode 负责
        tool_message_parser = StructuredOutputToToolMessageParser(
            llm=llm,
            enable_parallel_tool_calls=enable_parallel_tool_calls,
        )
        # 结构化输出模式：llm 输出经过 parser 转换为带 tool_calls 的 AIMessage
        chain = chat_prompt_template | llm | tool_message_parser
        # 添加工具描述到上下文变量
        context_variables = {
            **context_variables,
            "tools": render_text_description_and_args(list(tools)),
            "tool_names": ",".join([t.name for t in tools]),
        }
    else:
        # 原生工具调用模式：使用 bind_tools 绑定工具
        llm_with_tools = llm.bind_tools(tools) if tools else llm
        chain = chat_prompt_template | llm_with_tools

    return chain, context_variables


class ModelState(TypedDict, total=False):
    """模型节点的类型安全状态定义。

    使用 TypedDict 提供类型安全的状态访问，与 LangChain AgentState 模式保持一致。

    Attributes:
        messages: 消息列表，使用 add_messages 注解确保消息正确合并
    """

    messages: Required[Annotated[list[AnyMessage], add_messages]]


def build_model_node(
    *,
    llm: BaseChatModel,
    context_processor: ContextProcessor,
    node_options: ModelNodeSettings | None = None,
) -> RunnableCallable:
    """创建标准 QA 场景的模型节点。

    Args:
        llm: 语言模型
        context_processor: 上下文处理器实例（包含工具列表）
        node_options: 模型节点选项（可选，不传则使用默认值）

    Returns:
        RunnableCallable，包含同步和异步两个节点函数
    """

    if node_options is None:
        node_options = ModelNodeSettings()

    use_structured_response = node_options.use_structured_response
    enable_parallel_tool_calls = node_options.enable_parallel_tool_calls

    def model_node(
        state: ModelState,
        config: RunnableConfig,
        *,
        store: BaseStore,
    ) -> Dict[str, Any]:
        """
        同步模型推理节点。

        - 根据 state 中的 decision 选择合适的 ReAct Prompt
        - 支持两种模式：
          - use_structured_response=True: 使用结构化输出 + tool_message_parser
          - use_structured_response=False: 使用原生 function calling (bind_tools)

        Args:
            state: 类型安全的状态字典
            config: Runnable 配置
            store: LangGraph Store

        Returns:
            包含 messages 的字典
        """
        chain, context_variables = _prepare_model_chain(
            llm=llm,
            context_processor=context_processor,
            use_structured_response=use_structured_response,
            enable_parallel_tool_calls=enable_parallel_tool_calls,
            state=state,
            config=config,
            store=store,
        )
        # 调用模型
        response = chain.invoke(context_variables, config=config)
        return {"messages": [response]}

    async def amodel_node(
        state: ModelState,
        config: RunnableConfig,
        *,
        store: BaseStore,
    ) -> Dict[str, Any]:
        """
        异步模型推理节点。

        - 根据 state 中的 decision 选择合适的 ReAct Prompt
        - 支持两种模式：
          - use_structured_response=True: 使用结构化输出 + tool_message_parser
          - use_structured_response=False: 使用原生 function calling (bind_tools)

        Args:
            state: 类型安全的状态字典
            config: Runnable 配置
            store: LangGraph Store

        Returns:
            包含 messages 的字典
        """
        chain, context_variables = _prepare_model_chain(
            llm=llm,
            context_processor=context_processor,
            use_structured_response=use_structured_response,
            enable_parallel_tool_calls=enable_parallel_tool_calls,
            state=state,
            config=config,
            store=store,
        )
        # 异步调用模型
        response = await chain.ainvoke(context_variables, config=config)
        return {"messages": [response]}

    return RunnableCallable(model_node, amodel_node, trace=True)
