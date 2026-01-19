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

工具结果压缩函数模块 - 业务逻辑层

本模块提供工具结果压缩的业务逻辑实现（LLM Prompt、分块、工厂函数等）。
可作为 `aidev_agent.packages.langchain_core.tools.enhance.EnhancedTool` 的
`compressor_func` 传入，实现“执行后处理增强”。
"""

import json
import logging
from enum import Enum
from typing import Any, Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

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
    """将工具结果格式化为文本"""

    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list)):
        try:
            return json.dumps(result, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(result)
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
    *,
    invoke_intent: Optional[str] = None,
) -> str:
    """策略一：直接 LLM 压缩"""

    tool_intent = tool_intent or invoke_intent

    prompt = ChatPromptTemplate.from_messages(
        [("system", LLM_COMPRESSION_SYS_PROMPT), ("human", LLM_COMPRESSION_USR_PROMPT)], template_format="jinja2"
    )
    messages = prompt.format_messages(
        max_length=max_length,
        tool_name=tool_name,
        tool_description=tool_description,
        tool_intent=tool_intent,
        original_result=result_text,
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
    *,
    invoke_intent: Optional[str] = None,
    chunk_size: int = 120000,
    chunk_overlap: int = 200,
) -> str:
    """处理超长文本的LLM压缩策略"""

    tool_intent = tool_intent or invoke_intent

    # 分块算法：正确处理块的重叠
    texts = []
    for i in range(0, len(result_text), chunk_size):
        end = min(i + chunk_size + chunk_overlap, len(result_text))
        texts.append(result_text[i:end])

    res = [
        llm_compressor(
            text,
            llm,
            max_length,
            tool_name=tool_name,
            tool_description=tool_description,
            tool_intent=tool_intent,
        )
        for text in texts
    ]
    if len(res) == 1:
        return res[0]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", LLM_COMPRESSION_SYS_PROMPT),
            ("human", "由于工具返回结果过多，以下是经过处理后的结果，合并这些工具调用结果：{{result}}"),
        ],
        template_format="jinja2",
    )
    messages = prompt.format_messages(
        max_length=max_length,
        tool_name=tool_name,
        tool_description=tool_description,
        tool_intent=tool_intent,
        result="".join(f"<RESULT>以下是处理后的工具调用内容：{i}</RESULT>" for i in res),
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
) -> Callable[..., str]:
    """创建一个默认的压缩函数 - 业务逻辑层"""

    # 输入参数验证
    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    if llm is None:
        raise ValueError("llm parameter is required for compression")

    def custom_compressor(
        result: Any,
        tool_name: str,
        *,
        tool_description: Optional[str] = None,
        invoke_intent: Optional[str] = None,
        tool_intent: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """根据配置的策略对工具返回结果进行压缩"""

        effective_intent = tool_intent or invoke_intent

        # 统一处理结果转换为文本
        result_text = _format_result_to_text(result)
        if len(result_text) <= max_length:
            return result_text

        logger.debug(f"Compressing result with strategy {strategy}, original length: {len(result_text)}")

        # 根据不同的策略调用不同的压缩实现
        match strategy:
            case CompressionStrategy.SIMPLE:
                return llm_compressor_with_long(
                    result_text,
                    llm,
                    max_length=max_length,
                    tool_name=tool_name,
                    tool_description=tool_description,
                    tool_intent=effective_intent,
                    **kwargs,
                )
        raise ValueError(f"Unknown compression strategy: {strategy}")

    return custom_compressor
