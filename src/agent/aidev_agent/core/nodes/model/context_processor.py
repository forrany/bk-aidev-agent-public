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

import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from aidev_agent.core.graphs.react.prompts import DEFAULT_QA_PROMPT_TEMPLATES

from .basic_middleware import (
    BaseTemplateMiddleware,
    BaseToolsMiddleware,
    BaseVariablesMiddleware,
    DecisionBasedTemplateMiddleware,
    DeepSeekR1VariablesMiddleware,
    SpecialVariablesMiddleware,
)
from .pydantic_models import Middleware, ProcessorContext
from .token_compression import (
    ChatHistoryCompressionMiddleware,
    CompressionState,
    KnowledgeCompressionMiddleware,
    ToolOutputCompressionMiddleware,
)

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore


class MiddlewarePipeline:
    """类似 Koa/Express 的同步中间件执行链。"""

    def __init__(self) -> None:
        self._middlewares: list[Middleware] = []

    @property
    def middlewares(self) -> list[Middleware]:
        return list(self._middlewares)

    def use(self, middleware: Middleware, *, prepend: bool = False) -> None:
        if prepend:
            self._middlewares.insert(0, middleware)
        else:
            self._middlewares.append(middleware)

    def execute(self, ctx: ProcessorContext) -> None:
        middlewares = self._middlewares

        def dispatch(idx: int) -> None:
            if idx >= len(middlewares):
                return

            called = False

            def next_() -> None:
                nonlocal called
                if called:
                    raise RuntimeError("next() called multiple times")
                called = True
                dispatch(idx + 1)

            middlewares[idx](ctx, next_)

        dispatch(0)


class ContextProcessor:
    """上下文处理器：通过三条 MiddlewarePipeline 组织工具/模板/变量构建逻辑。"""

    chat_prompt_templates: ClassVar[Dict[str, Any]] = DEFAULT_QA_PROMPT_TEMPLATES

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
        enable_parallel_tool_calls: bool = True,
        tool_output_compress_thrd: int = int(os.getenv("TOOL_OUTPUT_COMPRESS_THRD", "5000")),
        tool_output_compressor_type: str = "specific",
    ):
        self.use_structured_response = use_structured_response
        self.enable_query_clarification = enable_query_clarification
        self.rejection_message = rejection_message
        self.role_prompt = role_prompt
        self.use_general_knowledge_on_miss = use_general_knowledge_on_miss
        self.prefix = prefix
        self.use_deepseek_r1_models_process = use_deepseek_r1_models_process
        self.enable_parallel_tool_calls = enable_parallel_tool_calls

        self._all_tools: List[BaseTool] = tools or []
        self.tool_output_compress_thrd = tool_output_compress_thrd
        self.tool_output_compressor_type = tool_output_compressor_type
        self._compression_state: CompressionState = CompressionState()
        self._cache: Dict[str, Any] = {}

        # ===== Middlewares =====
        self._tool_pipeline = MiddlewarePipeline()
        self._tool_pipeline.use(BaseToolsMiddleware())

        self._template_pipeline = MiddlewarePipeline()
        self._template_pipeline.use(
            BaseTemplateMiddleware(
                use_structured_response=self.use_structured_response,
                prefix=self.prefix,
                role_prompt=self.role_prompt,
            )
        )
        self._template_pipeline.use(
            DecisionBasedTemplateMiddleware(
                chat_prompt_templates=self.chat_prompt_templates,
                use_structured_response=self.use_structured_response,
                enable_query_clarification=self.enable_query_clarification,
            )
        )

        self._variable_pipeline = MiddlewarePipeline()
        self._variable_pipeline.use(BaseVariablesMiddleware())
        self._variable_pipeline.use(
            ToolOutputCompressionMiddleware(
                tool_output_compress_thrd=self.tool_output_compress_thrd,
                compressor_type=self.tool_output_compressor_type,
            )
        )
        self._variable_pipeline.use(
            SpecialVariablesMiddleware(
                use_structured_response=self.use_structured_response,
                use_general_knowledge_on_miss=self.use_general_knowledge_on_miss,
                rejection_message=self.rejection_message,
                role_prompt=self.role_prompt,
                enable_parallel_tool_calls=self.enable_parallel_tool_calls,
            )
        )
        self._variable_pipeline.use(DeepSeekR1VariablesMiddleware())
        self._variable_pipeline.use(KnowledgeCompressionMiddleware())
        self._variable_pipeline.use(ChatHistoryCompressionMiddleware())

    # ---------------------------------------------------------------------
    # Middleware injection
    # ---------------------------------------------------------------------

    def add_middleware(self, pipeline: str, middleware: Middleware, *, prepend: bool = False) -> None:
        """向指定 pipeline 注入自定义中间件。

        pipeline 可选值："tool"/"template"/"variable"（或其复数形式）。
        """

        key = pipeline.lower()
        if key in {"tool", "tools"}:
            self._tool_pipeline.use(middleware, prepend=prepend)
        elif key in {"template", "templates"}:
            self._template_pipeline.use(middleware, prepend=prepend)
        elif key in {"variable", "variables"}:
            self._variable_pipeline.use(middleware, prepend=prepend)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")

    def use_middleware(self, pipeline: str, middleware: Middleware, *, prepend: bool = False) -> None:
        """add_middleware 的别名。"""

        self.add_middleware(pipeline, middleware, prepend=prepend)

    def add_tool_middleware(self, middleware: Middleware, *, prepend: bool = False) -> None:
        self.add_middleware("tool", middleware, prepend=prepend)

    def add_template_middleware(self, middleware: Middleware, *, prepend: bool = False) -> None:
        self.add_middleware("template", middleware, prepend=prepend)

    def add_variable_middleware(self, middleware: Middleware, *, prepend: bool = False) -> None:
        self.add_middleware("variable", middleware, prepend=prepend)

    # ---------------------------------------------------------------------
    # Public API (backwards compatible)
    # ---------------------------------------------------------------------

    def get_choice_tools(self, state: Dict[str, Any], config: RunnableConfig) -> List[BaseTool]:
        ctx = ProcessorContext(state=state, config=config)
        ctx.metadata["all_tools"] = self._all_tools

        self._tool_pipeline.execute(ctx)
        return ctx.tools

    def get_chat_prompt_template(
        self, state: Dict[str, Any], config: RunnableConfig, store: "BaseStore"
    ) -> ChatPromptTemplate:
        ctx = ProcessorContext(state=state, config=config, store=store)

        self._template_pipeline.execute(ctx)

        if ctx.prompt_template is None:
            # 理论上 BaseTemplateMiddleware 会保证总有值
            raise RuntimeError("prompt_template is not set")

        return ctx.prompt_template

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
        ctx = ProcessorContext(
            state=state,
            config=config,
            llm=llm,
            chat_prompt_template=chat_prompt_template,
            token_limit=token_limit,
            token_margin=token_margin,
        )

        ctx.metadata.update(context)
        ctx.metadata.setdefault("_compression_state", replace(self._compression_state))
        ctx.metadata.setdefault("_cache", self._cache)
        ctx.metadata.setdefault("use_deepseek_r1_models_process", self.use_deepseek_r1_models_process)

        self._variable_pipeline.execute(ctx)

        # 回写压缩状态
        self._compression_state = ctx.metadata.get("_compression_state", self._compression_state)
        # 回写消息切分缓存（实例级）
        self._cache = ctx.metadata.get("_cache", self._cache)

        return ctx.variables
