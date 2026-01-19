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

工具执行增强模块 - 技术实现层

本模块提供对工具执行结果的“后处理增强”能力，包括：
- EnhancedTool: 增强工具包装类（执行原工具后，对结果做二次处理）
- create_enhanced_tool: 创建增强工具的便利函数

注意：结果压缩策略的业务逻辑（包括 LLM 压缩、关键词压缩等）
已移至 aidev_agent.packages.langchain_core.tools.compressor_func 模块。
"""

import copy
import inspect
import logging
from typing import Any, Callable, Coroutine, Optional, Protocol, Type, Union

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import ArgsSchema, BaseTool
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)


class CompressorFunc(Protocol):
    """压缩函数的协议定义

    压缩函数接收原始结果和工具名称，返回压缩后的字符串结果。
    支持同步和异步两种实现方式。
    """

    def __call__(
        self, original_result: Any, tool_name: str, *, invoke_intent: Optional[str] = None, **kwargs
    ) -> str: ...


class AsyncCompressorFunc(Protocol):
    """异步压缩函数的协议定义"""

    async def __call__(
        self, original_result: Any, tool_name: str, *, invoke_intent: Optional[str] = None, **kwargs
    ) -> str: ...


def _build_extended_schema(
    original_schema: Optional[ArgsSchema],
    intent_description: str,
) -> Union[Type[BaseModel], dict]:
    """Build an extended schema with required 'invoke_intent' field.

    Args:
        original_schema: 原始的参数 schema
        intent_description: invoke_intent 字段的描述

    Returns:
        扩展后的 schema，包含 invoke_intent 字段（必填）
    """
    # 情况 1: 原始 schema 是 BaseModel
    if isinstance(original_schema, type) and issubclass(original_schema, BaseModel):
        fields = {name: (field.annotation, field) for name, field in original_schema.model_fields.items()}
        # invoke_intent 作为必填字段
        fields["invoke_intent"] = (str, Field(..., description=intent_description))
        return create_model(f"{original_schema.__name__}WithIntent", **fields)
    # 情况 2: 原始 schema 是 dict（MCP 导出的）
    elif isinstance(original_schema, dict):
        schema = copy.deepcopy(original_schema)
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        # invoke_intent 作为必填字段
        schema["properties"]["invoke_intent"] = {"type": "string", "description": intent_description}
        schema.setdefault("required", [])
        if "invoke_intent" not in schema["required"]:
            schema["required"].append("invoke_intent")
        return schema
    # 情况 3: 没有 schema（空工具）
    else:
        return {
            "title": "ToolArgs",
            "type": "object",
            "properties": {"invoke_intent": {"type": "string", "description": intent_description}},
            "required": ["invoke_intent"],
        }


class EnhancedTool(BaseTool):
    """
    带结果后处理能力的增强工具包装器

    这个类包装一个现有的工具，在执行后对其返回结果做二次处理（例如压缩、格式化等）。
    后处理函数需要由调用方显式提供。

    特性：
    - 支持同步和异步后处理函数
    - 后处理失败时可降级返回原始结果（通过 fallback_on_error 控制）
    - 支持 invoke_intent 参数用于传递调用意图（当 show_intent=True 时强制启用）
    """

    original_tool: BaseTool
    compressor_func: Callable[..., str] | Coroutine[Any, Any, str]
    show_intent: bool
    fallback_on_error: bool = True

    def __init__(
        self,
        original_tool: BaseTool,
        compressor_func: Callable[..., str] | Coroutine[Any, Any, str],
        show_intent: bool = False,
        intent_description: Optional[str] = None,
        fallback_on_error: bool = True,
        **kwargs,
    ):
        """
        初始化增强工具

        Args:
            original_tool: 原始工具
            compressor_func: 压缩函数（必须提供）
            show_intent: 是否启用意图参数，当为 True 时工具会强制要求 invoke_intent 参数
            intent_description: 意图参数的描述
            fallback_on_error: 压缩失败时是否降级返回原始结果，默认 True
            **kwargs: 其他参数传递给父类
        """
        # 处理 args_schema，如果需要显示意图则添加 invoke_intent 字段
        intent_description = intent_description or (
            "请简要描述调用该工具的意图说明：你为什么调用这个工具、希望通过调用得到什么结果、以及这次调用与当前任务的关系。"
            "例如：'提取日志以诊断节点异常'、'查询用户配置用于后续部署'、'验证接口返回数据是否正确'。"
        )
        args_schema = original_tool.args_schema
        if show_intent:
            args_schema = _build_extended_schema(
                original_tool.args_schema,
                intent_description=intent_description,
            )

        # 继承原工具的属性
        # 安全获取可能不存在的属性
        super().__init__(
            name=original_tool.name,
            description=original_tool.description,
            args_schema=args_schema,
            return_direct=getattr(original_tool, "return_direct", False),
            verbose=getattr(original_tool, "verbose", False),
            callbacks=getattr(original_tool, "callbacks", None),
            tags=getattr(original_tool, "tags", None),
            metadata=getattr(original_tool, "metadata", None),
            handle_tool_error=getattr(original_tool, "handle_tool_error", False),
            original_tool=original_tool,
            compressor_func=compressor_func,
            show_intent=show_intent,
            fallback_on_error=fallback_on_error,
            **kwargs,
        )

    def _extract_invoke_intent(self, kwargs: dict) -> Optional[str]:
        """统一提取 invoke_intent 的逻辑

        Args:
            kwargs: 函数参数字典

        Returns:
            提取出的 invoke_intent，或 None 如果不需要

        Raises:
            ValueError: 如果 show_intent=True 但缺少 invoke_intent 参数
        """
        if not self.show_intent:
            # 即使不显示意图，也要清理可能存在的 invoke_intent 参数
            kwargs.pop("invoke_intent", None)
            return None

        invoke_intent = kwargs.pop("invoke_intent", None)

        # show_intent=True 时强制要求 invoke_intent
        if not invoke_intent:
            raise ValueError(
                f"invoke_intent is required for tool {self.original_tool.name} but not provided. "
                f"Please provide a brief description of why you are calling this tool."
            )
        return invoke_intent

    def _run(
        self,
        *args: Any,
        config: RunnableConfig,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        """
        执行工具并压缩结果
        """
        # 提取 invoke_intent 参数（如果存在）
        invoke_intent = self._extract_invoke_intent(kwargs)

        # 执行原工具
        logger.debug(f"执行原工具: {self.original_tool.name}")
        original_result = self.original_tool._run(*args, config=config, run_manager=run_manager, **kwargs)

        # 压缩结果，如果有意图信息则传递给压缩函数
        return self._compress_result(original_result, invoke_intent)

    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        """
        异步执行工具并压缩结果
        """
        # 提取 invoke_intent 参数（如果存在）
        invoke_intent = self._extract_invoke_intent(kwargs)

        # 执行原工具
        logger.debug(f"Executing original tool async: {self.original_tool.name}")
        original_result = await self.original_tool._arun(*args, config=config, run_manager=run_manager, **kwargs)

        # 异步压缩结果
        return await self._compress_result_async(original_result, invoke_intent)

    def _compress_result(self, original_result: Any, invoke_intent: Optional[str]) -> str:
        """
        同步压缩结果，支持失败降级

        Args:
            original_result: 原始工具返回结果
            invoke_intent: 工具调用意图

        Returns:
            压缩后的结果字符串
        """
        logger.debug(f"工具 {self.original_tool.name} 将执行压缩")
        try:
            compressed_result = self.compressor_func(
                original_result, tool_name=self.original_tool.name, invoke_intent=invoke_intent
            )
            logger.debug(f"工具 {self.original_tool.name} 结果压缩成功")
            return compressed_result
        except Exception as e:
            return self._handle_compression_error(e, original_result)

    async def _compress_result_async(self, original_result: Any, invoke_intent: Optional[str]) -> str:
        """
        异步压缩结果，支持失败降级

        Args:
            original_result: 原始工具返回结果
            invoke_intent: 工具调用意图

        Returns:
            压缩后的结果字符串
        """
        logger.debug(f"工具 {self.original_tool.name} 将执行压缩")
        try:
            if inspect.iscoroutinefunction(self.compressor_func):
                compressed_result = await self.compressor_func(
                    original_result, tool_name=self.original_tool.name, invoke_intent=invoke_intent
                )
            else:
                compressed_result = self.compressor_func(
                    original_result, tool_name=self.original_tool.name, invoke_intent=invoke_intent
                )
            logger.debug(f"工具 {self.original_tool.name} 结果压缩成功")
            return compressed_result
        except Exception as e:
            return self._handle_compression_error(e, original_result)

    def _handle_compression_error(self, error: Exception, original_result: Any) -> str:
        """
        处理压缩错误，支持降级返回原始结果

        Args:
            error: 压缩过程中的异常
            original_result: 原始工具返回结果

        Returns:
            降级后的结果字符串（如果启用降级）

        Raises:
            Exception: 如果未启用降级，则重新抛出原始异常
        """
        if self.fallback_on_error:
            logger.warning(f"工具 {self.original_tool.name} 压缩失败: {error}. [降级处理]: 输出原始结果")
            return original_result
        else:
            logger.warning(f"工具 {self.original_tool.name} 压缩失败: {error}.")
            raise


def create_enhanced_tool(
    original_tool: BaseTool,
    compressor_func: Callable[..., str] | Coroutine[Any, Any, str],
    show_intent: bool = False,
    intent_description: Optional[str] = None,
    fallback_on_error: bool = True,
    **kwargs,
) -> EnhancedTool:
    """
    创建增强工具的便利函数

    Args:
        original_tool: 原始工具
        compressor_func: 压缩函数（必须提供，不使用默认策略），支持同步和异步
        show_intent: 是否启用意图参数，当为 True 时工具会强制要求 invoke_intent 参数
        intent_description: 意图描述
        fallback_on_error: 压缩失败时是否降级返回原始结果，默认 True
        **kwargs: 其他参数

    Returns:
        增强后的工具

    Raises:
        ValueError: 如果 compressor_func 为 None
    """
    return EnhancedTool(
        original_tool=original_tool,
        compressor_func=compressor_func,
        show_intent=show_intent,
        intent_description=intent_description,
        fallback_on_error=fallback_on_error,
        **kwargs,
    )
