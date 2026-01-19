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

import json
import logging
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pytz
from langchain_community.adapters.openai import convert_dict_to_message, convert_message_to_dict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from aidev_agent.core.graphs.react.prompts import MULTI_MODAL_PREFIX, general_qa_prompt_structured_chat
from aidev_agent.enums import ContextType, Decision
from aidev_agent.packages.langchain_core.models.utils import is_deepseek_r1_series_models

from .pydantic_models import DEFAULT_ENABLE_PARALLEL_TOOL_CALLS, NextFunction, ProcessorContext

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions (from variables.py)
# =============================================================================


def get_beijing_now() -> str:
    """获取北京时间的格式化字符串。"""

    utc_now = datetime.now(pytz.utc)
    beijing_now = utc_now.astimezone(pytz.timezone("Asia/Shanghai")).strftime("%Y年%m月%d日 %H时%M分%S秒")
    return beijing_now


def get_context_type_from_state(state: Dict[str, Any]) -> str:
    knowledge_content = state.get("knowledge_content")
    knowledge_qa_content = state.get("knowledge_qa_content")

    if knowledge_content and knowledge_qa_content:
        return ContextType.BOTH.value
    elif knowledge_content:
        return ContextType.PRIVATE.value
    elif knowledge_qa_content:
        return ContextType.QA_RESPONSE.value
    return ""


def extract_tool_calls_from_messages(tool_messages: List[BaseMessage]) -> str:
    """将 AIMessage.tool_calls + ToolMessage 结果拼接为 structured agent_scratchpad 字符串。"""

    if not tool_messages:
        return ""

    tool_results: Dict[str, ToolMessage] = {}
    for msg in tool_messages:
        if isinstance(msg, ToolMessage):
            tool_results[msg.tool_call_id] = msg

    parts: list[str] = []
    for msg in tool_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_input = tool_call.get("args", {})
                tool_call_id = tool_call.get("id", "")

                observation = ""
                if tool_call_id and tool_call_id in tool_results:
                    observation = tool_results[tool_call_id].content

                tried_tool = json.dumps({"action": tool_name, "action_input": tool_input}, ensure_ascii=False)
                parts.append(f"\n已经调用过的工具：\n{tried_tool}")
                parts.append(f"\n观察结果：{observation}\n")

    return "".join(parts)


# =============================================================================
# Helper Functions (from template.py)
# =============================================================================


def create_tool_call_prompt_template(
    prefix: Optional[str] = None,
    role_prompt: Optional[str] = None,
    *,
    query_knowledgebase: bool = False,
) -> ChatPromptTemplate:
    """构造 Tool-Calling 场景下使用的 ChatPromptTemplate。"""

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
    """构造 Structured Chat 场景下使用的 ChatPromptTemplate。"""

    return deepcopy(general_qa_prompt_structured_chat)


# =============================================================================
# Tools Middleware (from tools.py)
# =============================================================================


class BaseToolsMiddleware:
    """设置初始工具列表。

    约定：ContextProcessor 在 ctx.metadata["all_tools"] 中提供全量工具列表。
    """

    def __call__(self, ctx: ProcessorContext, next: NextFunction) -> None:
        all_tools = ctx.metadata.get("all_tools", [])
        ctx.tools = list(all_tools)
        next()


class DecisionBasedToolFilterMiddleware:
    """根据 decision 过滤工具（可选）。

    约定：
    - ctx.metadata["tools_allowlist_by_decision"]: Dict[Decision, Set[str]] 或 Dict[str, Set[str]]
    """

    def __call__(self, ctx: ProcessorContext, next: NextFunction) -> None:
        decision = ctx.state.get("decision")
        allowlist_by_decision: Optional[Dict[Any, Set[str]]] = ctx.metadata.get("tools_allowlist_by_decision")

        if not allowlist_by_decision:
            next()
            return

        allow = allowlist_by_decision.get(decision)
        if allow is None and isinstance(decision, Decision):
            allow = allowlist_by_decision.get(decision.value)

        if allow:
            ctx.tools = [t for t in ctx.tools if isinstance(t, BaseTool) and t.name in allow]

        next()


# =============================================================================
# Template Middleware (from template.py)
# =============================================================================


class BaseTemplateMiddleware:
    """设置默认模板（非 QA 分支时使用）。"""

    def __init__(
        self,
        *,
        use_structured_response: bool,
        prefix: Optional[str],
        role_prompt: str,
    ):
        self.use_structured_response = use_structured_response
        self.prefix = prefix
        self.role_prompt = role_prompt

    def __call__(self, ctx: ProcessorContext, next: NextFunction) -> None:
        if self.use_structured_response:
            ctx.prompt_template = create_structured_chat_prompt_template()
        else:
            ctx.prompt_template = create_tool_call_prompt_template(prefix=self.prefix, role_prompt=self.role_prompt)

        next()


class DecisionBasedTemplateMiddleware:
    """根据 decision 选择模板（覆盖默认模板）。"""

    def __init__(
        self,
        *,
        chat_prompt_templates: Dict[str, Any],
        use_structured_response: bool,
        enable_query_clarification: bool,
    ):
        self.chat_prompt_templates = chat_prompt_templates
        self.use_structured_response = use_structured_response
        self.enable_query_clarification = enable_query_clarification

    def __call__(self, ctx: ProcessorContext, next: NextFunction) -> None:
        decision = ctx.state.get("decision", Decision.GENERAL_QA)
        suffix = "_structured_chat" if self.use_structured_response else "_tool_calling"

        template: Optional[ChatPromptTemplate]
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
            template = None

        if template is not None:
            ctx.prompt_template = template

        next()


# =============================================================================
# Variables Middleware (from variables.py)
# =============================================================================


class BaseVariablesMiddleware:
    """从 state 提取基础变量，并做消息拆分（支持缓存优化）。

    缓存策略：只缓存 last_human_idx，不缓存消息列表本身。
    通过切片 messages[:idx+1] 和 messages[idx+1:] 获取结果。
    """

    def __call__(self, ctx: ProcessorContext, next: NextFunction) -> None:
        if ctx.chat_prompt_template is None:
            next()
            return

        # 自动提取模板所需变量
        auto_vars: Dict[str, Any] = {}
        for var in ctx.chat_prompt_template.input_variables:
            if var in ctx.state:
                auto_vars[var] = ctx.state[var]

        messages: List[BaseMessage] = ctx.state.get("messages") or []
        cache = ctx.metadata.get("_cache")

        # 尝试使用缓存的 last_human_idx
        if isinstance(cache, dict) and self._is_cache_valid(cache, messages):
            last_human_idx = cache["last_human_idx"]
        else:
            last_human_idx = self._find_last_human_idx(messages)
            if isinstance(cache, dict):
                cache["messages_id"] = id(messages)
                cache["last_human_idx"] = last_human_idx

        # 基于 last_human_idx 切分消息
        if last_human_idx == -1:
            chat_history, tool_messages = [], list(messages)
        else:
            chat_history = list(messages[: last_human_idx + 1])
            tool_messages = list(messages[last_human_idx + 1 :])

        ctx.variables = {**ctx.variables, **auto_vars}
        ctx.metadata["chat_history"] = chat_history
        ctx.metadata["tool_messages"] = tool_messages

        next()

    @staticmethod
    def _find_last_human_idx(messages: List[BaseMessage]) -> int:
        """从尾部查找最后一条 HumanMessage/SystemMessage 的索引。"""
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], (HumanMessage, SystemMessage)):
                return i
        return -1

    @staticmethod
    def _is_cache_valid(cache: Dict[str, Any], messages: List[BaseMessage]) -> bool:
        """检查缓存是否有效：同一 messages 对象且最后一条消息不是 Human/System。"""
        if cache.get("messages_id") != id(messages):
            return False
        if not messages:
            return True
        # 只要最后一条消息不是 HumanMessage/SystemMessage，last_human_idx 就不会变
        return not isinstance(messages[-1], (HumanMessage, SystemMessage))


class SpecialVariablesMiddleware:
    """构建特殊变量（beijing_now/context_type/context 等）。"""

    def __init__(
        self,
        *,
        use_structured_response: bool,
        use_general_knowledge_on_miss: bool,
        rejection_message: str,
        role_prompt: str,
        enable_parallel_tool_calls: bool = DEFAULT_ENABLE_PARALLEL_TOOL_CALLS,
    ):
        self.use_structured_response = use_structured_response
        self.use_general_knowledge_on_miss = use_general_knowledge_on_miss
        self.rejection_message = rejection_message
        self.role_prompt = role_prompt
        self.enable_parallel_tool_calls = enable_parallel_tool_calls

    def __call__(self, ctx: ProcessorContext, next: NextFunction) -> None:
        chat_history: List[BaseMessage] = ctx.metadata.get("chat_history", [])
        tool_messages: List[BaseMessage] = ctx.metadata.get("tool_messages", [])

        if self.use_structured_response:
            agent_scratchpad: Any = extract_tool_calls_from_messages(tool_messages)
        else:
            agent_scratchpad = tool_messages

        special_vars = {
            "beijing_now": get_beijing_now(),
            "context_type": get_context_type_from_state(ctx.state),
            "context": ctx.state.get("knowledge_content"),
            "qa_context": ctx.state.get("knowledge_qa_content"),
            "query": chat_history[-1].content if chat_history else "",
            "use_general_knowledge_on_miss": self.use_general_knowledge_on_miss,
            "chat_history": chat_history[:-1],
            "rejection_response": self.rejection_message,
            "role_prompt": self.role_prompt,
            "agent_scratchpad": agent_scratchpad,
            "enable_parallel_tool_calls": self.enable_parallel_tool_calls,
        }

        ctx.variables = {**ctx.variables, **special_vars}

        next()


class DeepSeekR1VariablesMiddleware:
    """DeepSeek-R1：避免使用 system prompt（将 SystemMessage 视为 user）。"""

    def __call__(self, ctx: ProcessorContext, next: NextFunction) -> None:
        llm = ctx.llm
        if llm is None or not ctx.metadata.get("use_deepseek_r1_models_process", True):
            next()
            return
        is_r1 = is_deepseek_r1_series_models(llm)
        if not is_r1:
            next()
            return
        chat_history = ctx.variables.get("chat_history")
        if isinstance(chat_history, list):
            for i in range(len(chat_history)):
                if isinstance(chat_history[i], SystemMessage):
                    msg = convert_message_to_dict(chat_history[i])
                    msg["role"] = "user"
                    chat_history[i] = convert_dict_to_message(msg)
        next()
