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
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Optional, Tuple

import pytz
from langchain_community.adapters.openai import convert_message_to_dict, convert_dict_to_message
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from aidev_agent.enums import ContextType, Decision
from aidev_agent.packages.langchain_core.models.utils import is_deepseek_r1_series_models
from aidev_agent.core.graphs.react.prompts import (
    DEFAULT_QA_PROMPT_TEMPLATES,
    MULTI_MODAL_PREFIX,
    general_qa_prompt_structured_chat,
)
from aidev_agent.packages.langgraph.streaming.utils import conditional_dispatch_custom_event

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

logger = logging.getLogger(__name__)


def create_tool_call_prompt_template(
    prefix: Optional[str] = None,
    role_prompt: Optional[str] = None,
    *,
    query_knowledgebase: bool = False,
) -> ChatPromptTemplate:
    """构造 Tool-Calling 场景下使用的 ChatPromptTemplate。

    逻辑参考 ToolCallCommonAgentMixIn.create_agent：
    - system: 多模态前缀 + 角色提示
    - placeholder: chat_history
    - human: 当前用户输入
    - placeholder: agent_scratchpad
    - 可选插入知识库查询的提示语
    """
    messages = [
        (
            "system",
            (prefix or MULTI_MODAL_PREFIX) + ("\n" + role_prompt if role_prompt else "") + "\n",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
    if query_knowledgebase:
        messages.insert(
            -2,
            (
                "human",
                "根据后续用户提的问题，获取knowledge_item_ids与knowledgebase_ids, 先使用工具查询下知识库。"
                "如果发现knowledge_items或knowledgebase都和主题无关，那就随机挑选一个存在的。",
            ),
        )
        messages.insert(
            -2,
            (
                "ai",
                "好的，接下来我会先查询下知识库，并确保传入了knowledge_item_ids或knowledgebase_ids。",
            ),
        )
    return ChatPromptTemplate.from_messages(messages)


def create_structured_chat_prompt_template() -> ChatPromptTemplate:
    """构造 Structured Chat 场景下使用的 ChatPromptTemplate。

    逻辑参考 StructuredChatCommonAgentMixIn.create_agent：
    - 直接复用 general_qa_prompt_structured_chat，并进行 deepcopy，避免被运行时修改。
    """
    return deepcopy(general_qa_prompt_structured_chat)


def get_beijing_now() -> str:
    """获取北京时间的格式化字符串。"""
    utc_now = datetime.now(pytz.utc)
    beijing_now = utc_now.astimezone(pytz.timezone("Asia/Shanghai")).strftime("%Y年%m月%d日 %H时%M分%S秒")
    return beijing_now


# ===== 压缩策略类型定义 =====
CompressionStrategy = Callable[[Dict[str, Any], Optional[BaseChatModel], Dict[str, Any]], Tuple[bool, Dict[str, Any]]]


def compress_knowledge_context(
    variables: Dict[str, Any],
    llm: Optional[BaseChatModel],
    context: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """
    压缩知识库内容策略（优先级1）

    Args:
        variables: 当前变量字典
        llm: 语言模型（用于压缩）
        context: 压缩上下文，包含压缩状态等信息

    Returns:
        (是否执行了压缩, 更新后的变量字典)
    """
    compression_state = context.get("_compression_state", {})

    if "context" in variables and variables["context"] and not compression_state.get("context_compressed", False):
        # 从 context 中获取 intent_recognition_instance
        intent_recognition_instance = context.get("intent_recognition_instance")
        if not intent_recognition_instance:
            logger.debug("intent_recognition_instance 未在 context 中提供，跳过知识库内容压缩")
            return False, variables

        provided_chat_history = context.get("provided_chat_history", [])

        # 发送压缩日志事件
        conditional_dispatch_custom_event(
            "custom_event",
            {"compress_log": "\n```text\nToken 超限，尝试压缩知识库知识内容以减少 token 使用。\n```\n"},
        )

        variables["context"] = intent_recognition_instance.llm_context_compressor_parallel(
            provided_chat_history,
            variables.get("query", ""),
            variables["context"],
            llm,
        )

        compression_state["context_compressed"] = True
        context["_compression_state"] = compression_state
        return True, variables

    return False, variables


def compress_chat_history(
    variables: Dict[str, Any],
    llm: Optional[BaseChatModel],
    context: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """
    压缩对话历史策略（优先级2）：依次移除最早的对话记录

    Args:
        variables: 当前变量字典
        llm: 语言模型（未使用，保持接口一致）
        context: 压缩上下文

    Returns:
        (是否执行了压缩, 更新后的变量字典)
    """
    compression_state = context.get("_compression_state", {})

    if "chat_history" in variables and variables["chat_history"]:
        # 第一次进入此策略时发送日志
        if not compression_state.get("chat_history_compression_started", False):
            conditional_dispatch_custom_event(
                "custom_event",
                {"compress_log": "\n```text\nToken 超限，尝试抛除会话历史以减少 token 使用。\n```\n"},
            )
            compression_state["chat_history_compression_started"] = True
            context["_compression_state"] = compression_state

        # 移除最早的一条消息
        variables["chat_history"] = variables["chat_history"][1:]
        compression_state["chat_history_compression_count"] = (
            compression_state.get("chat_history_compression_count", 0) + 1
        )
        context["_compression_state"] = compression_state
        return True, variables

    return False, variables


# 默认压缩策略列表
DEFAULT_COMPRESSION_STRATEGIES: List[CompressionStrategy] = [
    compress_knowledge_context,
    compress_chat_history,
]


class ContextProcessor:
    """
    上下文处理器，负责管理模型节点所需的上下文逻辑。

    职责：
    - 计算 context_type（根据 state 中的知识内容）
    - 选择合适的 prompt 模板（根据 decision 类型）
    - 准备传入 LLM 的上下文变量
    - 管理消息压缩（可选）

    类变量：
    - chat_prompt_templates: 包含所有支持的 prompt 模板，子类可覆盖自定义
    - compression_strategies: 默认压缩策略列表，子类可覆盖自定义
    """

    # 类变量：prompt 模板字典
    chat_prompt_templates: ClassVar[Dict[str, Any]] = DEFAULT_QA_PROMPT_TEMPLATES

    # 类变量：默认压缩策略
    compression_strategies: ClassVar[List[CompressionStrategy]] = DEFAULT_COMPRESSION_STRATEGIES

    def __init__(
        self,
        *,
        use_structured_response: bool,
        enable_query_clarification: bool,
        rejection_message: str,
        role_prompt: str,
        use_general_knowledge_on_miss: bool,
        prefix: Optional[str] = None,
        use_deepseek_r1_models_process: bool = True,
        tools: Optional[List[BaseTool]] = None,
    ):
        """
        初始化 ContextProcessor。

        Args:
            use_structured_response: 是否使用 structured chat 模式
            enable_query_clarification: 是否启用查询澄清
            rejection_message: 拒答消息
            role_prompt: 角色提示
            use_general_knowledge_on_miss: 知识库未命中时是否使用通用知识
            prefix: 系统提示前缀（用于 tool_calling 模式）
            use_deepseek_r1_models_process: 是否使用 deepseek r1 系列模型处理
            tools: 工具列表，用于 get_choice_tools 方法
        """
        self.use_structured_response = use_structured_response
        self.enable_query_clarification = enable_query_clarification
        self.rejection_message = rejection_message
        self.role_prompt = role_prompt
        self.use_general_knowledge_on_miss = use_general_knowledge_on_miss
        self.prefix = prefix
        self._all_tools: List[BaseTool] = tools or []

        # 压缩状态管理
        self._compression_state: Dict[str, Any] = {
            "context_compressed": False,
            "chat_history_compression_started": False,
            "chat_history_compression_count": 0,
        }
        self.use_deepseek_r1_models_process = use_deepseek_r1_models_process

    def get_context_type(self, state: Dict[str, Any]) -> str:
        """
        根据 state 计算 context_type。

        Args:
            state: 包含 knowledge_content 和 knowledge_qa_content 的状态字典

        Returns:
            context_type 字符串值
        """
        knowledge_content = state.get("knowledge_content")
        knowledge_qa_content = state.get("knowledge_qa_content")

        if knowledge_content and knowledge_qa_content:
            return ContextType.BOTH.value
        elif knowledge_content:
            return ContextType.PRIVATE.value
        elif knowledge_qa_content:
            return ContextType.QA_RESPONSE.value
        return ""

    def get_chat_prompt_template(
        self,
        state: Dict[str, Any],
        config: RunnableConfig,
        store: "BaseStore",
    ) -> ChatPromptTemplate:
        """
        根据 state 中的 decision 选择对应的 prompt 模板。

        从类变量 chat_prompt_templates 中获取模板，对于需要动态填充的模板，
        使用 partial 填充 prefix 和 role_prompt。

        Args:
            state: 状态字典，包含 decision 等信息
            config: Runnable 配置
            store: LangGraph Store

        Returns:
            对应的 ChatPromptTemplate，已通过 partial 填充动态变量
        """
        # 从 state 中获取 decision，默认为 GENERAL_QA
        decision = state.get("decision", Decision.GENERAL_QA)

        suffix = "_structured_chat" if self.use_structured_response else "_tool_calling"

        if decision == Decision.GENERAL_QA:
            template = self.chat_prompt_templates.get(f"general_qa_prompt{suffix}")
        elif decision == Decision.PRIVATE_QA:
            template = self.chat_prompt_templates.get(f"private_qa_prompt{suffix}")
        elif decision == Decision.QUERY_CLARIFICATION:
            if self.enable_query_clarification:
                template = self.chat_prompt_templates.get(f"clarifying_qa_prompt{suffix}")
            else:
                template = self.chat_prompt_templates.get(f"private_qa_prompt{suffix}")
        else:
            # 默认返回 react_chat_prompt_template
            if self.use_structured_response:
                template = create_structured_chat_prompt_template()
            else:
                template = create_tool_call_prompt_template(prefix=self.prefix, role_prompt=self.role_prompt)

        # 返回模板（如果需要可以在这里添加 partial 逻辑）
        return template

    def get_choice_tools(
        self,
        state: Dict[str, Any],
        config: RunnableConfig,
    ) -> List[BaseTool]:
        """
        根据运行时状态选择要绑定的工具。

        默认返回所有配置的工具，子类可覆盖以实现动态工具选择。

        Args:
            state: 状态字典
            config: Runnable 配置

        Returns:
            选择的工具列表
        """
        return self._all_tools

    def _split_messages(
        self, messages: List[BaseMessage]
    ) -> Tuple[List[BaseMessage], List[BaseMessage]]:
        """
        将 messages 切割为 chat_history 和 agent_scratchpad。

        切割逻辑：
        - 找到最后一个 HumanMessage 的位置
        - chat_history: 最后一个 HumanMessage 之前的所有消息（包含该 HumanMessage）
        - agent_scratchpad: 最后一个 HumanMessage 之后的所有消息（AIMessage、ToolMessage 等）

        这样做是为了保持与现有 prompt 模板的兼容性，它们依赖 agent_scratchpad 占位符。
        agent_scratchpad 语义上表示本轮对话中的工具调用交互。

        Args:
            messages: 完整的消息列表

        Returns:
            (chat_history, agent_scratchpad) 元组
        """
        if not messages:
            return [], []

        # 从后往前找最后一个 HumanMessage 的位置
        last_human_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage) or isinstance(messages[i], SystemMessage):
                last_human_idx = i
                break

        # 如果没有找到 HumanMessage，将所有消息作为 chat_history
        if last_human_idx == -1:
            return [], list(messages)

        # 切割：chat_history 包含最后一个 HumanMessage，agent_scratchpad 是之后的消息
        chat_history = list(messages[: last_human_idx + 1])
        agent_scratchpad = list(messages[last_human_idx + 1:])

        return chat_history, agent_scratchpad

    def _build_special_variables(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建特殊变量（需要额外处理的变量）。

        Args:
            state: 状态字典

        Returns:
            特殊变量字典
        """
        # 从 messages 中切割出 chat_history 和 agent_scratchpad
        messages = state.get("messages", [])
        chat_history, agent_scratchpad = self._split_messages(messages)

        return {
            "beijing_now": get_beijing_now(),
            "context_type": self.get_context_type(state),
            "context": state.get("knowledge_content"),
            "qa_context": state.get("knowledge_qa_content"),
            "query": state.get("input"),
            "use_general_knowledge_on_miss": self.use_general_knowledge_on_miss,
            "chat_history": chat_history,
            "rejection_response": self.rejection_message,
            "role_prompt": self.role_prompt,
            "agent_scratchpad": agent_scratchpad,
        }

    def get_chat_prompt_variables(
        self,
        *,
        chat_prompt_template: ChatPromptTemplate,
        state: Dict[str, Any],
        config: RunnableConfig,
        llm: Optional[BaseChatModel] = None,
        token_limit: Optional[int] = None,
        token_margin: int = 100,
        **context,
    ) -> Dict[str, Any]:
        """
        准备传入 LLM chain 的上下文变量。

        从 state 中提取模板所需的变量，并构建特殊变量。
        如果提供了 token_limit，还会执行压缩逻辑以确保不超过 token 限制。

        Args:
            chat_prompt_template: ChatPromptTemplate 模板
            state: 状态字典
            config: Runnable 配置
            llm: 语言模型（用于压缩时计算 token）
            token_limit: token 限制（可选，提供时启用压缩）
            token_margin: token 余量
            **context: 额外上下文，例如：
                - intent_recognition_instance: 意图识别实例（用于知识库压缩）
                - provided_chat_history: 对话历史

        Returns:
            包含所有 LLM 所需变量的字典
        """
        # 1. 自动从 state 中提取模板所需的变量
        auto_vars = {}
        for var in chat_prompt_template.input_variables:
            if var in state:
                auto_vars[var] = state[var]

        # 2. 构建特殊变量
        messages = state.get("messages", [])
        chat_history, agent_scratchpad = self._split_messages(messages)
        # 根据 deepseek 官方建议 https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#usage-recommendations
        # deepseek-r1 系列模型需要避免使用 system prompt
        # 这里统一转一下（否则用户选择"预设角色"可能包含 system prompt）
        # NOTE: 虽然聊天窗侧统一支持了以下转换，但还需要支持插件侧使用，因此这里还是需要做下检测和转换
        if isinstance(llm, BaseChatModel) and is_deepseek_r1_series_models(llm):
            for i in range(len(chat_history)):
                if isinstance(chat_history[i], SystemMessage):
                    msg = convert_message_to_dict(chat_history[i])
                    msg["role"] = "user"
                    chat_history[i] = convert_dict_to_message(msg)

        special_vars = {
            "beijing_now": get_beijing_now(),
            "context_type": self.get_context_type(state),
            "context": state.get("knowledge_content"),
            "qa_context": state.get("knowledge_qa_content"),
            "query": state.get("input"),
            "use_general_knowledge_on_miss": self.use_general_knowledge_on_miss,
            "chat_history": chat_history,
            "rejection_response": self.rejection_message,
            "role_prompt": self.role_prompt,
            "agent_scratchpad": agent_scratchpad,
        }
        # 3. 合并，特殊变量优先
        variables = {**auto_vars, **special_vars}

        # 4. 如果提供了 token_limit 和 llm，执行压缩逻辑
        if token_limit is not None and llm is not None:
            variables = self._ensure_token_limit(
                variables=variables,
                chat_prompt_template=chat_prompt_template,
                llm=llm,
                token_limit=token_limit,
                token_margin=token_margin,
                **context,
            )

        return variables

    def _ensure_token_limit(
        self,
        *,
        variables: Dict[str, Any],
        chat_prompt_template: ChatPromptTemplate,
        llm: BaseChatModel,
        token_limit: int,
        token_margin: int,
        **context,
    ) -> Dict[str, Any]:
        """
        确保变量不超过 token 限制，必要时执行压缩策略。

        Args:
            variables: 当前变量字典
            chat_prompt_template: ChatPromptTemplate 模板
            llm: 语言模型
            token_limit: token 限制
            token_margin: token 余量
            **context: 额外上下文

        Returns:
            符合 token 限制的变量字典
        """
        # 将压缩状态添加到 context
        context["_compression_state"] = self._compression_state

        # 计算当前 token 数
        try:
            formatted_prompt = chat_prompt_template._format_prompt_with_error_handling(variables)
            cur_token_len = llm.get_num_tokens_from_messages(formatted_prompt.messages)
        except Exception as e:
            logger.warning(f"计算 token 长度失败: {e}")
            return variables

        while cur_token_len > token_limit - token_margin:
            compressed = False

            # 按顺序尝试压缩策略
            for strategy in self.compression_strategies:
                executed, variables = strategy(variables, llm, context)
                if executed:
                    compressed = True
                    break

            if not compressed:
                # 所有策略都无法压缩，记录警告并返回当前变量
                err_msg = (
                    "已尝试按优先级压缩上下文，但还是超过 token 限制。"
                    f"（当前 token 数: {cur_token_len}，限制: {token_limit}，余量: {token_margin}）"
                )
                logger.warning(err_msg)
                break

            # 重新计算 token 数
            try:
                formatted_prompt = chat_prompt_template._format_prompt_with_error_handling(variables)
                cur_token_len = llm.get_num_tokens_from_messages(formatted_prompt.messages)
            except Exception as e:
                logger.warning(f"计算 token 长度失败: {e}")
                break

        # 更新实例的压缩状态
        self._compression_state = context.get("_compression_state", self._compression_state)

        return variables

    def reset_compression_state(self) -> None:
        """重置压缩状态，用于新一轮对话。"""
        self._compression_state = {
            "context_compressed": False,
            "chat_history_compression_started": False,
            "chat_history_compression_count": 0,
        }

    # 向后兼容：保留旧方法名作为别名
    def prepare_context_variables(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备传入 LLM chain 的上下文变量（向后兼容方法）。

        已废弃，请使用 get_chat_prompt_variables。

        Args:
            state: 状态字典

        Returns:
            包含所有 LLM 所需变量的字典
        """
        return self._build_special_variables(state)


__all__ = [
    "ContextProcessor",
    "get_beijing_now",
    "create_tool_call_prompt_template",
    "create_structured_chat_prompt_template",
    "CompressionStrategy",
    "compress_knowledge_context",
    "compress_chat_history",
    "DEFAULT_COMPRESSION_STRATEGIES",
]
