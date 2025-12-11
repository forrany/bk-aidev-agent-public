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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore


class NextFunction(Protocol):
    def __call__(self) -> None: ...


@dataclass
class ProcessorContext:
    """中间件共享上下文。

    - 输入：state/config/store/llm 等
    - 输出：tools/prompt_template/variables
    - metadata：中间件间共享的扩展数据
    """

    # 输入
    state: Dict[str, Any]
    config: RunnableConfig
    store: Optional["BaseStore"] = None
    llm: Optional[BaseChatModel] = None
    chat_prompt_template: Optional[ChatPromptTemplate] = None
    token_limit: Optional[int] = None
    token_margin: int = 100

    # 输出
    tools: List[BaseTool] = field(default_factory=list)
    prompt_template: Optional[ChatPromptTemplate] = None
    variables: Dict[str, Any] = field(default_factory=dict)

    # 中间件间通信
    metadata: Dict[str, Any] = field(default_factory=dict)


class Middleware(Protocol):
    def __call__(self, ctx: ProcessorContext, next: NextFunction) -> None: ...


# =============================================================================
# Settings (from settings.py)
# =============================================================================


DEFAULT_ENABLE_PARALLEL_TOOL_CALLS: bool = True


class ModelNodeSettings(BaseModel):
    """Settings for `build_model_node`."""

    use_structured_response: bool = False
    enable_parallel_tool_calls: bool = True
