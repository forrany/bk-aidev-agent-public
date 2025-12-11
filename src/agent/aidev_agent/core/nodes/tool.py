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
import json
import logging
import time
from collections.abc import Awaitable, Sequence
from enum import Enum
from typing import Any, Optional, Callable, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolCallWrapper, AsyncToolCallWrapper
from langgraph.types import Command

logger = logging.getLogger(__name__)


class CompressionStrategy(str, Enum):
    """压缩策略枚举"""
    SIMPLE = "simple"
    ITERATIVE_KEYWORD = "iterative_keyword"


# ============================================================================
# 压缩相关的 Prompt 模板 - 业务逻辑
# ============================================================================

LLM_COMPRESSION_SYS_PROMPT = """
你是一个专业的信息压缩专家，请根据工具调用意图对以下工具执行结果进行智能压缩。

**核心目标：** 在不超过 {{max_length}} 个字符的限制下，提取与用户原始请求最相关、最有价值的信息。

**压缩规则：**
1. **以意图为核心：** 优先保留与**用户原始请求**直接相关的数据、状态、操作结果或错误原因。
2.  **区分结果类型：**
    *   **正常输出：** 筛选核心数据（ID、名称、状态、关键数值、返回对象）、成功提示、最终状态或重要返回值。
    *   **错误/异常：** 重点提取错误码、错误消息、导致错误的原因（例如：参数错误、资源不存在、权限不足）和可能的解决建议。
    *   **日志/冗余信息：** 仅保留关键操作记录、最终结果反馈，忽略调试信息、性能指标或重复内容。
3.  **格式要求：**
    *   使用简洁的自然语言。
    *   保持准确性，不得引入错误或修改原始含义。
    *   使用客观性描述，避免主动评价或建议，避免过度概括或推测。
    *   避免寒暄和解释性语句。
    *   禁止生成新的标题、章节编号、分析、解释或总结性语句。
    *   最后输出"完成"来结束。
"""

LLM_COMPRESSION_USR_PROMPT = """
{% if tool_name %}
工具名称: {{tool_name}}
{% endif %}
{% if tool_description %}
工具描述: {{tool_description}}
{% endif %}
{% if tool_intent %}
工具意图: {{tool_intent}}
{% endif %}
原始结果：```
{{original_result}}
```

**注意核心目标：提取与用户原始请求最相关、最有价值的信息！使用客观性描述，避免主动评价或建议，避免过度概括或推测。**
"""


# ============================================================================
# 工具结果格式化函数
# ============================================================================

def _format_result_to_text(result: Any) -> str:
    """
    将工具结果格式化为文本

    Args:
        result: 工具执行结果

    Returns:
        格式化后的文本
    """
    if isinstance(result, str):
        return result
    elif isinstance(result, (dict, list)):
        try:
            return json.dumps(result, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(result)
    else:
        return str(result)


# ============================================================================
# 压缩策略实现 - 业务逻辑
# ============================================================================

def llm_compressor(
    result_text: str,
    llm: BaseChatModel,
    max_length: int,
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
    tool_intent: Optional[str] = None,
) -> str:
    """
    策略一：直接 LLM 压缩

    Args:
        result_text: 原始结果文本
        llm: 语言模型
        max_length: 目标压缩长度
        tool_name: 工具名称
        tool_description: 工具描述
        tool_intent: 工具调用意图

    Returns:
        压缩后的文本
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", LLM_COMPRESSION_SYS_PROMPT),
        ("human", LLM_COMPRESSION_USR_PROMPT)
    ], template_format="jinja2")
    messages = prompt.format_messages(
        max_length=max_length,
        tool_name=tool_name,
        tool_description=tool_description,
        tool_intent=tool_intent,
        original_result=result_text
    )
    response = llm.invoke(messages)
    return response.content.strip()


def llm_compressor_with_long(
    result_text: str,
    llm: BaseChatModel,
    max_length: int,
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
    tool_intent: Optional[str] = None,
    chunk_size: int = 120000,
    chunk_overlap: int = 200,
) -> str:
    """
    处理超长文本的LLM压缩策略

    Args:
        result_text: 原始结果文本
        llm: 语言模型
        max_length: 目标压缩长度
        tool_name: 工具名称
        tool_description: 工具描述
        tool_intent: 工具调用意图
        chunk_size: 分块大小
        chunk_overlap: 块之间的重叠长度

    Returns:
        压缩后的文本
    """
    # 分块算法：正确处理块的重叠
    texts = []
    for i in range(0, len(result_text), chunk_size):
        end = min(i + chunk_size + chunk_overlap, len(result_text))
        texts.append(result_text[i:end])

    res = [
        llm_compressor(text, llm, max_length, tool_name, tool_description, tool_intent)
        for text in texts
    ]
    if len(res) == 1:
        return res[0]

    prompt = ChatPromptTemplate.from_messages([
        ("system", LLM_COMPRESSION_SYS_PROMPT),
        ("human", "由于工具返回结果过多，以下是经过处理后的结果，合并这些工具调用结果：{{result}}")
    ], template_format="jinja2")
    messages = prompt.format_messages(
        max_length=max_length,
        tool_name=tool_name,
        tool_description=tool_description,
        tool_intent=tool_intent,
        result="".join(f"<RESULT>以下是处理后的工具调用内容：{i}</RESULT>" for i in res)
    )
    logger.debug(f"Merging {len(res)} compressed results for tool: {tool_name}")
    response = llm.invoke(messages)
    return response.content.strip()


# ============================================================================
# 压缩器工厂函数 - 业务逻辑入口
# ============================================================================

def create_default_compressor(
    strategy: CompressionStrategy = CompressionStrategy.SIMPLE,
    llm: Optional[BaseChatModel] = None,
    max_length: int = 500,
    max_iterations: int = 5,
) -> Callable[..., str]:
    """
    创建一个默认的压缩函数 - 业务逻辑层

    这个函数负责根据业务配置创建压缩函数。

    Args:
        strategy: 压缩策略 (SIMPLE 或 ITERATIVE_KEYWORD)
        llm: 用于压缩的语言模型（必须提供）
        max_length: 压缩后的最大长度 (某些策略需要)
        max_iterations: 最大迭代次数 (某些策略需要)

    Returns:
        一个配置好的压缩函数

    Raises:
        ValueError: 如果 max_length <= 0 或 max_iterations <= 0 或 llm 为 None
    """
    # 输入参数验证
    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be > 0")
    if llm is None:
        raise ValueError("llm parameter is required for compression")

    def custom_compressor(result: Any, **kwargs) -> str:
        """
        根据配置的策略对工具返回结果进行压缩

        Args:
            result: 工具执行结果
            **kwargs: 其他参数，包括 tool_name、tool_intent 等

        Returns:
            压缩后的结果文本
        """
        # 统一处理结果转换为文本
        result_text = _format_result_to_text(result)
        if len(result_text) <= max_length:
            return result_text

        logger.debug(f"Compressing result with strategy {strategy}, original length: {len(result_text)}")

        # 根据不同的策略调用不同的压缩实现
        match strategy:
            case CompressionStrategy.SIMPLE:
                return llm_compressor_with_long(result_text, llm, max_length=max_length, **kwargs)
        raise ValueError(f"Unknown compression strategy: {strategy}")

    return custom_compressor


# ============================================================================
# 工具节点中间件相关函数
# ============================================================================
def _chain_tool_call_wrappers(
    wrappers: Sequence[ToolCallWrapper],
) -> ToolCallWrapper | None:
    """将多个工具调用包装器组合为中间件栈（第一个为最外层）。

    Args:
        wrappers: 按中间件顺序排列的包装器序列。

    Returns:
        组合后的包装器，如果列表为空则返回 None。

    Example:
        wrapper = _chain_tool_call_wrappers([auth, cache, retry])
        # 请求流向: auth -> cache -> retry -> tool
        # 响应流向: tool -> retry -> cache -> auth
    """
    if not wrappers:
        return None

    if len(wrappers) == 1:
        return wrappers[0]

    def compose_two(outer: ToolCallWrapper, inner: ToolCallWrapper) -> ToolCallWrapper:
        """组合两个包装器，outer 包装 inner。"""

        def composed(
            request: ToolCallRequest,
            execute: Callable[[ToolCallRequest], ToolMessage | Command],
        ) -> ToolMessage | Command:
            # 创建一个可调用对象，使用原始 execute 调用 inner
            def call_inner(req: ToolCallRequest) -> ToolMessage | Command:
                return inner(req, execute)

            # outer 可以多次调用 call_inner
            return outer(request, call_inner)

        return composed

    # 链式组合所有包装器: first -> second -> ... -> last
    result = wrappers[-1]
    for wrapper in reversed(wrappers[:-1]):
        result = compose_two(wrapper, result)

    return result


def _chain_async_tool_call_wrappers(
    wrappers: Sequence[AsyncToolCallWrapper],
) -> AsyncToolCallWrapper | None:
    """将多个异步工具调用包装器组合为中间件栈（第一个为最外层）。

    Args:
        wrappers: 按中间件顺序排列的异步包装器序列。

    Returns:
        组合后的异步包装器，如果列表为空则返回 None。
    """
    if not wrappers:
        return None

    if len(wrappers) == 1:
        return wrappers[0]

    def compose_two(
        outer: AsyncToolCallWrapper,
        inner: AsyncToolCallWrapper,
    ) -> AsyncToolCallWrapper:
        """组合两个异步包装器，outer 包装 inner。"""

        async def composed(
            request: ToolCallRequest,
            execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
        ) -> ToolMessage | Command:
            # 创建一个异步可调用对象，使用原始 execute 调用 inner
            async def call_inner(req: ToolCallRequest) -> ToolMessage | Command:
                return await inner(req, execute)

            # outer 可以多次调用 call_inner
            return await outer(request, call_inner)

        return composed

    # 链式组合所有包装器: first -> second -> ... -> last
    result = wrappers[-1]
    for wrapper in reversed(wrappers[:-1]):
        result = compose_two(wrapper, result)

    return result


def _add_metadata_to_tool_message(
    msg: ToolMessage,
    duration: int,
    description: str | None,
) -> None:
    """为 ToolMessage 添加执行元数据。

    Args:
        msg: 要添加元数据的 ToolMessage
        duration: 执行时长（毫秒）
        description: 工具描述
    """
    msg.additional_kwargs.setdefault("duration", duration)
    if description:
        msg.additional_kwargs.setdefault("description", description)


def timer_sync_wrapper(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """同步计时包装器：记录工具执行时长并添加工具描述到元数据。

    Args:
        request: 工具调用请求
        execute: 执行函数

    Returns:
        工具执行结果，附带 duration 和 description 元数据
    """
    t1 = int(time.time() * 1000)
    msg = execute(request)
    t2 = int(time.time() * 1000)
    duration = t2 - t1

    # 从 request.tool 获取工具描述
    description = None
    if request.tool:
        description = getattr(request.tool, "description", None)

    # 为 ToolMessage 添加元数据
    if isinstance(msg, ToolMessage):
        _add_metadata_to_tool_message(msg, duration, description)
    elif isinstance(msg, Command) and msg.update:
        # 处理 Command 中的 ToolMessage
        for each in msg.update.get("messages", []):
            if isinstance(each, ToolMessage):
                _add_metadata_to_tool_message(each, duration, description)

    return msg


async def timer_async_wrapper(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
) -> ToolMessage | Command:
    """异步计时包装器：记录工具执行时长并添加工具描述到元数据。

    Args:
        request: 工具调用请求
        execute: 异步执行函数

    Returns:
        工具执行结果，附带 duration 和 description 元数据
    """
    t1 = int(time.time() * 1000)
    msg = await execute(request)
    t2 = int(time.time() * 1000)
    duration = t2 - t1

    # 从 request.tool 获取工具描述
    description = None
    if request.tool:
        description = getattr(request.tool, "description", None)

    # 为 ToolMessage 添加元数据
    if isinstance(msg, ToolMessage):
        _add_metadata_to_tool_message(msg, duration, description)
    elif isinstance(msg, Command) and msg.update:
        # 处理 Command 中的 ToolMessage
        for each in msg.update.get("messages", []):
            if isinstance(each, ToolMessage):
                _add_metadata_to_tool_message(each, duration, description)

    return msg


def build_tool_node(
    tools: Sequence[BaseTool | Callable],
    *,
    name: str = "tools",
    tags: List[str] | None = None,
    handle_tool_errors: bool | str | Callable[..., str] | type[Exception] | tuple[type[Exception], ...] = True,
    messages_key: str = "messages",
    wrappers: Sequence[ToolCallWrapper] | None = None,
    async_wrappers: Sequence[AsyncToolCallWrapper] | None = None,
) -> ToolNode:
    """构造带有中间件支持的 ToolNode。

    该函数会自动添加计时包装器，为每次工具调用收集执行时长和工具描述，
    并将这些元数据添加到 ToolMessage 的 additional_kwargs 中。

    Args:
        tools: 可供此节点调用的工具序列。支持 BaseTool 实例和普通函数。
        name: 节点在图中的名称标识符。
        tags: 可选的元数据标签。
        handle_tool_errors: 错误处理配置。True 表示捕获所有错误并返回包含错误信息的 ToolMessage。
        messages_key: 状态字典中包含消息列表的键名。
        wrappers: 可选的自定义同步包装器列表，会在计时包装器之后执行。
        async_wrappers: 可选的自定义异步包装器列表，会在计时包装器之后执行。

    Returns:
        配置了中间件的 ToolNode 实例。

    Example:
        # 基本用法
        tool_node = build_tool_node(tools=[my_tool])

        # 添加自定义包装器
        def my_wrapper(request, execute):
            logger.info(f"Calling tool: {request.tool_call['name']}")
            return execute(request)

        tool_node = build_tool_node(tools=[my_tool], wrappers=[my_wrapper])

        # 忽略所有错误
        tool_node = build_tool_node(tools=[my_tool], ignore_errors=True)
    """

    # 组合同步包装器：计时包装器 + 用户自定义包装器
    sync_wrapper_list: list[ToolCallWrapper] = [timer_sync_wrapper]
    if wrappers:
        sync_wrapper_list.extend(wrappers)
    final_sync_wrapper = _chain_tool_call_wrappers(sync_wrapper_list)

    # 组合异步包装器：计时包装器 + 用户自定义包装器
    async_wrapper_list: list[AsyncToolCallWrapper] = [timer_async_wrapper]
    if async_wrappers:
        async_wrapper_list.extend(async_wrappers)
    final_async_wrapper = _chain_async_tool_call_wrappers(async_wrapper_list)

    return ToolNode(
        tools=tools,
        name=name,
        tags=tags,
        handle_tool_errors=handle_tool_errors,
        messages_key=messages_key,
        wrap_tool_call=final_sync_wrapper,
        awrap_tool_call=final_async_wrapper,
    )
