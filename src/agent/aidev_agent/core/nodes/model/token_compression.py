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

Token 压缩中间件

该模块负责：
- Token 超限检测
- 知识库内容压缩（带哈希缓存复用）
- 工具输出压缩（基于长度阈值 或 Token 超限触发）
- 聊天历史压缩（渐进式移除最早消息）

为保持向后兼容：
- 保留 TokenCompressionMiddleware / TokenOverflowMiddleware 别名
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from jinja2 import BaseLoader
from jinja2.sandbox import SandboxedEnvironment as Environment
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from aidev_agent.packages.langchain_core.models.utils import is_deepseek_r1_series_models, remove_thinking_process
from aidev_agent.packages.langchain_core.retrievers.utils import HUNYUAN_SPECIFIC_RESPONSE
from aidev_agent.packages.langgraph.streaming.utils import conditional_dispatch_custom_event

env = Environment(loader=BaseLoader)
from .pydantic_models import NextFunction, ProcessorContext

logger = logging.getLogger(__name__)

_compression_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.getenv("COMPRESSION_EXECUTOR_MAX_WORKERS", "10"))
)


# ============================================================================
# 压缩状态
# ============================================================================
@dataclass
class CompressionState:
    """压缩状态（类型安全）。

    - knowledge_*: 知识库压缩的哈希/缓存/是否已压缩
    - tool_output_*: 工具输出压缩状态
    - chat_history_*: 聊天历史累计移除条数 & 已处理消息数量基准线
    """

    # 知识库压缩状态
    knowledge_hash: Optional[str] = None
    knowledge_cache: Optional[str] = None
    knowledge_compressed: bool = False

    # 工具输出压缩状态
    tool_output_compressed: bool = False
    tool_output_compressed_ids: set = field(default_factory=set)

    # 聊天历史压缩状态
    chat_history_removed: int = 0
    chat_history_baseline: int = 0

    @classmethod
    def from_legacy(cls, raw: Dict[str, Any]) -> "CompressionState":
        """兼容旧版 Dict `_compression_state` 的迁移。"""
        knowledge_compressed = bool(raw.get("knowledge_compressed", raw.get("context_compressed", False)))
        chat_history_removed = int(raw.get("chat_history_removed", raw.get("chat_history_compression_count", 0)) or 0)
        chat_history_baseline = int(raw.get("chat_history_baseline", 0) or 0)
        tool_output_compressed = bool(raw.get("tool_output_compressed", False))
        tool_output_compressed_ids = set(raw.get("tool_output_compressed_ids", []))

        return cls(
            knowledge_hash=raw.get("knowledge_hash"),
            knowledge_cache=raw.get("knowledge_cache"),
            knowledge_compressed=knowledge_compressed,
            tool_output_compressed=tool_output_compressed,
            tool_output_compressed_ids=tool_output_compressed_ids,
            chat_history_removed=chat_history_removed,
            chat_history_baseline=chat_history_baseline,
        )


def _ensure_compression_state(metadata: Dict[str, Any]) -> CompressionState:
    raw = metadata.get("_compression_state")

    if isinstance(raw, CompressionState):
        return raw

    if isinstance(raw, dict):
        state = CompressionState.from_legacy(raw)
    else:
        state = CompressionState()

    metadata["_compression_state"] = state
    return state


# ============================================================================
# 基类
# ============================================================================
class BaseCompressionMiddleware:
    """压缩中间件基类：提供通用的 token 超限检测与事件发送。"""

    @staticmethod
    def _dispatch_log(ctx: ProcessorContext, *, text: str) -> None:
        conditional_dispatch_custom_event(
            "custom_event",
            {"compress_log": f"\n```text\n{text}\n```\n"},
            enable_custom_event=ctx.metadata.get("enable_custom_event", True),
        )

    @staticmethod
    def _try_get_token_len(ctx: ProcessorContext) -> Optional[int]:
        if ctx.chat_prompt_template is None or ctx.llm is None:
            return None

        try:
            formatted_prompt = ctx.chat_prompt_template._format_prompt_with_error_handling(ctx.variables)
            return ctx.llm.get_num_tokens_from_messages(formatted_prompt.messages)
        except Exception as e:
            logger.warning(f"计算 token 长度失败: {e}")
            return None

    @classmethod
    def _is_overflow(cls, ctx: ProcessorContext) -> bool:
        if ctx.token_limit is None:
            return False

        token_len = cls._try_get_token_len(ctx)
        if token_len is None:
            return False

        return token_len > ctx.token_limit - ctx.token_margin


# ============================================================================
# 知识库内容压缩
# ============================================================================
class KnowledgeCompressionMiddleware(BaseCompressionMiddleware):
    """知识库内容压缩中间件：通过内容哈希缓存复用，避免重复压缩。

    需要在 ctx.metadata 中提供:
    - knowledge_compressor_func: Callable[[list, str, Any, Any], Any] - 压缩函数
    - provided_chat_history: list - 用于压缩的聊天历史
    """

    @staticmethod
    def _compute_hash(content: Any) -> str:
        if content is None:
            return ""
        if not isinstance(content, str):
            content = repr(content)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def __call__(self, ctx: ProcessorContext, next: NextFunction) -> None:
        if ctx.chat_prompt_template is None or ctx.llm is None or ctx.token_limit is None:
            next()
            return

        if not ("context" in ctx.variables and ctx.variables.get("context")):
            next()
            return

        state = _ensure_compression_state(ctx.metadata)

        # 内容变更检测：一旦变更，清理旧缓存。
        cur_hash = self._compute_hash(ctx.variables.get("context"))
        if cur_hash and cur_hash != state.knowledge_hash:
            state.knowledge_hash = cur_hash
            state.knowledge_cache = None
            state.knowledge_compressed = False

        # 已压缩且命中缓存：后续 ReAct 循环直接复用缓存。
        if state.knowledge_compressed and state.knowledge_cache and cur_hash == state.knowledge_hash:
            ctx.variables["context"] = state.knowledge_cache

        if not self._is_overflow(ctx):
            next()
            return

        # 超限但已压缩过：直接交给下一个中间件（如聊天历史压缩）。
        if state.knowledge_compressed and state.knowledge_cache:
            next()
            return

        compressor_func: Optional[Callable] = ctx.metadata.get("knowledge_compressor_func")
        if not compressor_func:
            logger.debug("knowledge_compressor_func 未在 metadata 中提供，跳过知识库内容压缩")
            next()
            return

        provided_chat_history = ctx.metadata.get("provided_chat_history", [])

        self._dispatch_log(ctx, text="Token 超限，尝试压缩知识库知识内容以减少 token 使用。")

        compressed_context = compressor_func(
            provided_chat_history,
            ctx.variables.get("query", ""),
            ctx.variables["context"],
            ctx.llm,
        )

        state.knowledge_cache = compressed_context
        state.knowledge_compressed = True
        ctx.variables["context"] = compressed_context

        next()


# ============================================================================
# 聊天历史压缩
# ============================================================================
class ChatHistoryCompressionMiddleware(BaseCompressionMiddleware):
    """聊天历史压缩中间件：累积移除并渐进式执行，不修改原始 state.messages。"""

    def __call__(self, ctx: ProcessorContext, next: NextFunction) -> None:
        if ctx.chat_prompt_template is None or ctx.llm is None or ctx.token_limit is None:
            next()
            return

        state = _ensure_compression_state(ctx.metadata)

        chat_history = ctx.variables.get("chat_history")
        if not isinstance(chat_history, list) or not chat_history:
            next()
            return

        # 先应用历史累计移除量（确保跨 ReAct 循环可复用）。
        removed = max(0, int(state.chat_history_removed or 0))
        if removed:
            chat_history = list(chat_history)[removed:]
        else:
            chat_history = list(chat_history)

        ctx.variables["chat_history"] = chat_history

        messages_len = len(ctx.state.get("messages") or [])
        if state.chat_history_baseline <= 0:
            state.chat_history_baseline = messages_len

        if not self._is_overflow(ctx):
            # 未超限：更新基准线（表示已处理到当前消息数）。
            state.chat_history_baseline = messages_len
            next()
            return

        dispatched = False

        # 超限：逐条移除最早消息，直到不超限或无可移除。
        while self._is_overflow(ctx):
            cur_chat_history = ctx.variables.get("chat_history")
            if not isinstance(cur_chat_history, list) or not cur_chat_history:
                logger.warning(
                    "已尝试抛除会话历史，但仍然超过 token 限制。"
                    f"（限制: {ctx.token_limit}，余量: {ctx.token_margin}）"
                )
                break

            if not dispatched:
                self._dispatch_log(ctx, text="Token 超限，尝试抛除会话历史以减少 token 使用。")
                dispatched = True

            # 不修改原 list：始终生成新 list
            ctx.variables["chat_history"] = list(cur_chat_history)[1:]
            state.chat_history_removed += 1

        # 记录已处理消息基准线
        state.chat_history_baseline = messages_len

        next()


# ============================================================================
# 工具输出压缩 - Prompt 模板
# ============================================================================
_COMMON_COMPRESSOR_SYS_PROMPT = (
    "对提供给你的内容进行摘要总结，要求不能丢失关键信息。直接返回你总结后的摘要即可，不要返回其他任何内容！"
)
_COMMON_COMPRESSOR_USR_PROMPT = env.from_string("提供给你的内容如下：```{{content}}```")

_SPECIFIC_COMPRESSOR_SYS_PROMPT = env.from_string(
    """
你是一个工具调用结果相关性判断与摘要生成器。你的任务是判断 {{candidate_tool_name}} 工具的调用结果是否能够**部分或全部回答用户最新提问**。

请遵循以下规则：

1. **相关性判断标准**：
   - 只要工具结果中包含**可用于回答用户最新提问中任何一个子问题或信息点的内容**，无论信息是否完整、是否需要推理、是否隐藏在结构化数据中，都视为"可以回答"。
   - 允许通过**语义推断、数值计算、上下文关联或常识理解**从工具结果中得出答案，不要求原文与提问完全一致。

2. **摘要要求**：
   - 仅提取与用户最新提问直接相关的内容。
   - 摘要必须**言简意赅，保留回答所需的关键信息**（如数值、名称、时间、状态、因果关系等）。
   - 避免复制大段原始数据，优先提炼成自然语言短句。

3. **输出规则**：
   - 如果工具结果**能提供任何有助于回答提问的信息** → 返回**摘要**。
   - 只有当工具结果**完全不包含任何相关信息、或信息完全无法用于回答提问时** → 返回："无效的工具调用"。

4. **特别注意**：
   - 为了让你可以更好地理解用户最新提问，我还会提供给你一段会话历史以供参考，格式如下：[HumanMessage(content='xxx'), AIMessage(content='xxx'), ...]
     其中"HumanMessage"表示用户历史提问，"AIMessage"表示智能聊天系统的历史回答。
   - 会话历史仅用于帮助理解当前提问的上下文，你的判断对象是**用户最新提问**与**工具调用结果**之间的相关性。
   - 工具结果可能包含冗余、噪声或结构化字段（如JSON日志、API响应），请聚焦其中**潜在有用的部分**。
   - **宁可保留一条模糊但可能相关的信息，也不要轻易判定为"无效"**。

直接返回摘要内容或"无效的工具调用"，不要输出任何解释、前缀、格式标记或额外说明。
"""
)

_SPECIFIC_COMPRESSOR_USR_PROMPT = env.from_string(
    "提供给你参考的会话历史内容如下：```{{provided_chat_history}}```"
    "\n\n\n给你的 {{candidate_tool_name}} 工具的调用结果如下：```{{candidate_tool_result}}```"
    "\n\n\n用户最新提问如下：```{{query}}```"
)


# ============================================================================
# 工具输出压缩
# ============================================================================
class ToolOutputCompressionMiddleware(BaseCompressionMiddleware):
    """工具调用结果压缩中间件。

    支持两种触发方式：
    1. 工具输出字符长度超过阈值（独立检查）
    2. Token 超限时被协调中间件调用

    参数说明：
    - tool_output_compress_thrd: 工具输出字符长度阈值，超过则触发压缩
    - compressor_type: 压缩模式，"specific"（带上下文）或 "common"（简单总结）
    - max_retries: LLM 调用最大重试次数
    - enable_token_overflow_compression: 是否在 Token 超限时也触发压缩

    需要在 ctx.metadata 中提供:
    - tool_messages: list[BaseMessage] - 工具消息列表
    - provided_chat_history: list - 用于压缩的聊天历史（compressor_type="specific" 时）
    """

    def __init__(
        self,
        *,
        tool_output_compress_thrd: int = 5000,
        compressor_type: str = "specific",
        max_retries: int = 5,
        enable_token_overflow_compression: bool = True,
    ) -> None:
        self.tool_output_compress_thrd = tool_output_compress_thrd
        self.compressor_type = compressor_type
        self.max_retries = max_retries
        self.enable_token_overflow_compression = enable_token_overflow_compression

    # -------------------------------------------------------------------------
    # 辅助方法
    # -------------------------------------------------------------------------
    @staticmethod
    def _get_query(ctx: ProcessorContext) -> str:
        """获取用户 query。"""
        query = ctx.variables.get("query")
        if isinstance(query, str) and query:
            return query

        chat_history = ctx.metadata.get("chat_history")
        if isinstance(chat_history, list) and chat_history:
            last = chat_history[-1]
            content = getattr(last, "content", "")
            return content if isinstance(content, str) else str(content)

        raw = ctx.state.get("input", "")
        return raw if isinstance(raw, str) else str(raw)

    @staticmethod
    def _get_tool_messages(ctx: ProcessorContext) -> Optional[List[BaseMessage]]:
        """获取 tool_messages。"""
        tool_messages = ctx.metadata.get("tool_messages")
        if isinstance(tool_messages, list) and tool_messages:
            return tool_messages
        return None

    @staticmethod
    def _tool_output_len(tool_messages: List[BaseMessage]) -> int:
        """计算工具输出的总字符长度。"""
        parts: List[str] = []
        for m in tool_messages:
            if isinstance(m, ToolMessage):
                content = m.content
                if content is not None:
                    parts.append(content if isinstance(content, str) else str(content))
        return len("".join(parts))

    @staticmethod
    def _collect_tool_msg_positions(
        tool_messages: List[BaseMessage],
    ) -> List[Tuple[int, ToolMessage]]:
        """收集 ToolMessage 的索引位置。"""
        positions: List[Tuple[int, ToolMessage]] = []
        for idx, m in enumerate(tool_messages):
            if isinstance(m, ToolMessage):
                positions.append((idx, m))
        return positions

    def _invoke_llm(self, *, llm: Any, messages: List[BaseMessage], config: Any) -> str:
        """调用 LLM 并处理 DeepSeek-R1 的特殊情况。"""
        try:
            is_r1 = is_deepseek_r1_series_models(llm)
        except Exception:
            is_r1 = False

        call_messages = list(messages)
        # DeepSeek-R1：避免使用 system prompt（将 SystemMessage 合并到 HumanMessage）。
        if is_r1 and call_messages and isinstance(call_messages[0], SystemMessage):
            sys_content = call_messages[0].content
            if len(call_messages) >= 2 and isinstance(call_messages[-1], HumanMessage):
                human_content = call_messages[-1].content
                call_messages[-1] = HumanMessage(content=f"{sys_content}\n\n{human_content}")
                del call_messages[0]

        try:
            resp = llm.invoke(call_messages, config=config)
        except TypeError:
            resp = llm.invoke(call_messages)

        content = getattr(resp, "content", "")
        content_str = content if isinstance(content, str) else str(content)

        if is_r1:
            try:
                content_str = remove_thinking_process(content_str).strip()
            except Exception:
                content_str = content_str.strip()

        return content_str

    def _compress_single_tool_output(
        self,
        *,
        llm: Any,
        config: Any,
        provided_chat_history: Any,
        query: str,
        tool_name: str,
        tool_result: Any,
    ) -> str:
        """压缩单个工具的输出。"""
        tool_result_str = tool_result if isinstance(tool_result, str) else str(tool_result)

        if self.compressor_type == "common":
            sys_prompt = _COMMON_COMPRESSOR_SYS_PROMPT
            usr_prompt = _COMMON_COMPRESSOR_USR_PROMPT.render(content=tool_result_str)
        elif self.compressor_type == "specific":
            sys_prompt = _SPECIFIC_COMPRESSOR_SYS_PROMPT.render(candidate_tool_name=tool_name)
            usr_prompt = _SPECIFIC_COMPRESSOR_USR_PROMPT.render(
                provided_chat_history=provided_chat_history,
                query=query,
                candidate_tool_name=tool_name,
                candidate_tool_result=tool_result_str,
            )
        else:
            raise ValueError(f"不支持的工具调用结果压缩方式：{self.compressor_type}")

        messages: List[BaseMessage] = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=usr_prompt),
        ]

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp_content = self._invoke_llm(llm=llm, messages=messages, config=config)
                # 如果触发了混元的特殊回复，则不进行压缩
                if resp_content == HUNYUAN_SPECIFIC_RESPONSE:
                    logger.debug(f"工具 {tool_name} 触发混元特殊回复，不压缩")
                    return tool_result_str
                logger.debug(f"工具 {tool_name} 压缩完成: {len(tool_result_str)} -> {len(resp_content)}")
                return resp_content
            except Exception as e:
                last_err = e
                logger.warning(f"工具 {tool_name} 压缩失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")

        if last_err:
            logger.error(f"工具 {tool_name} 压缩最终失败: {last_err}")
        return tool_result_str

    # -------------------------------------------------------------------------
    # 核心压缩逻辑
    # -------------------------------------------------------------------------
    def _do_compress(
        self,
        ctx: ProcessorContext,
        tool_messages: List[BaseMessage],
        tool_msg_positions: List[Tuple[int, ToolMessage]],
        state: CompressionState,
        *,
        reason: str,
    ) -> bool:
        """执行实际的压缩操作。

        Args:
            ctx: 处理器上下文
            tool_messages: 工具消息列表
            tool_msg_positions: ToolMessage 的索引位置
            state: 压缩状态
            reason: 压缩原因（用于日志）

        Returns:
            是否执行了压缩
        """
        # 过滤出尚未压缩的 ToolMessage
        uncompressed_positions: List[Tuple[int, ToolMessage]] = []
        for pos, tool_msg in tool_msg_positions:
            tool_call_id = tool_msg.tool_call_id
            if tool_call_id and tool_call_id in state.tool_output_compressed_ids:
                continue  # 已压缩过，跳过
            uncompressed_positions.append((pos, tool_msg))

        if not uncompressed_positions:
            logger.debug("所有工具输出已压缩，跳过")
            return False

        provided_chat_history = ctx.metadata.get("provided_chat_history", [])
        query = self._get_query(ctx)

        self._dispatch_log(ctx, text=reason)

        # 并发压缩
        futures = {
            _compression_executor.submit(
                self._compress_single_tool_output,
                llm=ctx.llm,
                config=ctx.config,
                provided_chat_history=provided_chat_history,
                query=query,
                tool_name=(tool_msg.name or "unknown"),
                tool_result=tool_msg.content,
            ): i
            for i, (_, tool_msg) in enumerate(uncompressed_positions)
        }

        results: List[Optional[str]] = [None] * len(uncompressed_positions)
        try:
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    logger.warning(f"调用 LLM 来对工具调用结果进行压缩总结时失败，索引 {i}，错误：{e}")
        except Exception as e:
            logger.warning(f"调用 LLM 来对工具调用结果进行压缩总结时失败，错误：{e}")

        # 以新的 ToolMessage 列表回写，避免修改原始 state.messages。
        new_tool_messages: List[BaseMessage] = list(tool_messages)
        for i, (pos, tool_msg) in enumerate(uncompressed_positions):
            compressed = results[i]
            if compressed is None:
                continue

            new_tool_messages[pos] = ToolMessage(
                content=compressed,
                name=tool_msg.name,
                tool_call_id=tool_msg.tool_call_id,
            )

            # 记录已压缩的 tool_call_id
            if tool_msg.tool_call_id:
                state.tool_output_compressed_ids.add(tool_msg.tool_call_id)

        ctx.metadata["tool_messages"] = new_tool_messages
        state.tool_output_compressed = True

        return True

    # -------------------------------------------------------------------------
    # 两种触发方式
    # -------------------------------------------------------------------------
    def compress_if_too_long(self, ctx: ProcessorContext) -> bool:
        """基于字符长度的压缩（独立检查）。

        当工具输出总字符长度超过 tool_output_compress_thrd 时触发。

        Returns:
            是否执行了压缩
        """
        if ctx.llm is None:
            return False

        tool_messages = self._get_tool_messages(ctx)
        if not tool_messages:
            return False

        if self._tool_output_len(tool_messages) <= self.tool_output_compress_thrd:
            return False

        tool_msg_positions = self._collect_tool_msg_positions(tool_messages)
        if not tool_msg_positions:
            return False

        state = _ensure_compression_state(ctx.metadata)

        return self._do_compress(
            ctx,
            tool_messages,
            tool_msg_positions,
            state,
            reason="工具调用结果过长，尝试压缩工具调用结果以减少 token 使用。",
        )

    def compress_for_token_overflow(self, ctx: ProcessorContext) -> bool:
        """Token 超限时触发的压缩。

        当 Token 超限且 enable_token_overflow_compression=True 时触发。

        Returns:
            是否执行了压缩
        """
        if not self.enable_token_overflow_compression:
            return False

        if ctx.llm is None:
            return False

        tool_messages = self._get_tool_messages(ctx)
        if not tool_messages:
            return False

        tool_msg_positions = self._collect_tool_msg_positions(tool_messages)
        if not tool_msg_positions:
            return False

        state = _ensure_compression_state(ctx.metadata)

        # 如果所有工具输出都已压缩过，不重复压缩
        all_compressed = all(
            (tool_msg.tool_call_id and tool_msg.tool_call_id in state.tool_output_compressed_ids)
            for _, tool_msg in tool_msg_positions
        )
        if all_compressed:
            return False

        return self._do_compress(
            ctx,
            tool_messages,
            tool_msg_positions,
            state,
            reason="Token 超限，尝试压缩工具调用结果以减少 token 使用。",
        )

    # -------------------------------------------------------------------------
    # 中间件入口
    # -------------------------------------------------------------------------
    def __call__(self, ctx: ProcessorContext, next: NextFunction) -> None:
        """中间件入口：仅处理基于长度的压缩。

        Token 超限的压缩由 TokenCompressionMiddleware 协调调用 compress_for_token_overflow。
        """
        self.compress_if_too_long(ctx)
        next()


# ============================================================================
# 统一协调中间件
# ============================================================================
class TokenCompressionMiddleware:
    """统一的 Token 压缩协调中间件。

    按优先级处理：
    1. 工具输出压缩（基于长度阈值，与 token 无关）
    2. 知识库内容压缩（Token 超限时）
    3. 工具输出压缩（Token 超限时）
    4. 聊天历史压缩（Token 超限时）

    参数说明：
    - tool_output_compress_thrd: 工具输出字符长度阈值
    - compressor_type: 压缩模式，"specific" 或 "common"
    """

    def __init__(
        self,
        *,
        tool_output_compress_thrd: int = 5000,
        compressor_type: str = "specific",
    ) -> None:
        self._knowledge = KnowledgeCompressionMiddleware()
        self._tool_output = ToolOutputCompressionMiddleware(
            tool_output_compress_thrd=tool_output_compress_thrd,
            compressor_type=compressor_type,
            enable_token_overflow_compression=True,
        )
        self._chat_history = ChatHistoryCompressionMiddleware()

    def __call__(self, ctx: ProcessorContext, next: NextFunction) -> None:
        # Step 1: 基于长度的工具输出压缩（独立于 token 检查）
        self._tool_output.compress_if_too_long(ctx)

        # Step 2-4: 按优先级处理 Token 超限
        def _after_knowledge() -> None:
            # 优先级 2: 工具输出压缩（Token 超限）
            if self._tool_output._is_overflow(ctx):
                self._tool_output.compress_for_token_overflow(ctx)

            # 优先级 3: 聊天历史压缩
            self._chat_history(ctx, next)

        # 优先级 1: 知识库压缩
        self._knowledge(ctx, _after_knowledge)


# 向后兼容：旧命名别名
TokenOverflowMiddleware = TokenCompressionMiddleware
