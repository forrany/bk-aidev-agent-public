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

import dataclasses
import json
import logging
import os
import traceback
from enum import Enum
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.trace import Span
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ExporterType(Enum):
    """OTEL Exporter 类型"""

    GRPC = "grpc"
    HTTP = "http"


class CallbackFilteredJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, dict) and "callbacks" in o:
            del o["callbacks"]
            return o

        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)

        if hasattr(o, "to_json"):
            return o.to_json()

        if isinstance(o, BaseModel) and hasattr(o, "model_dump_json"):
            return o.model_dump_json()

        return super().default(o)


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.

    @param func: The function to wrap
    @return: The wrapper function
    """
    # Obtain a logger specific to the function's module
    logger = logging.getLogger(func.__module__)

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.debug(
                "OpenLLMetry failed to trace in %s, error: %s",
                func.__name__,
                traceback.format_exc(),
            )

    return wrapper


def _safe_attach_context(span: Span):
    """
    安全地将 span 附加到 context,处理异步场景下的潜在失败

    Args:
        span: 要附加的 Span

    Returns:
        context token 用于后续 detach,失败时返回 None
    """
    try:
        # 使用 context_api.attach 而不是 _RUNTIME_CONTEXT.attach
        # 这样可以确保 context 正确传播到 LangChain 的自动插桩中
        return context_api.attach(trace.set_span_in_context(span))
    except Exception as e:
        logger.warning(f"Context attach failed, span hierarchy may be incorrect: {e}")
        return None


def _safe_detach_context(token):
    """
    安全地分离 context token,不会导致应用崩溃

    此方法实现了一个故障安全的 context 分离,处理异步/并发场景中
    context token 可能失效的所有已知边缘情况

    Args:
        token: context token
    """
    if not token:
        return

    try:
        # 直接使用 runtime context 避免 context_api.detach() 的错误日志
        # context_api 会使用 logger.exception 记录
        # 根据 LangChain 官方的说法，LangChain detach 失败是安全的
        from opentelemetry.context import _RUNTIME_CONTEXT

        _RUNTIME_CONTEXT.detach(token)
    except Exception as e:
        # Context detach 在异步场景下可能失败,这是预期的行为
        # 常见场景:
        # 1. Token 在一个 async task/thread 中创建,在另一个中 detach
        # 2. Context 已经被其他进程 detach
        # 3. Token 由于 context 切换而失效
        # 4. 高并发场景下的竞态条件
        #
        # 这是安全的,因为 span 本身已经正确结束,追踪数据已正确捕获
        logger.debug(f"Context detach failed: {e}")


def _set_span_attribute(span: Span, key: str, value: Any) -> None:
    value = _sanitize_metadata_value(value)
    if value is not None:
        if value != "":
            span.set_attribute(key, value)
        else:
            span.set_attribute(key, "")


def _sanitize_metadata_value(value: Any) -> Any:
    """Convert metadata values to OpenTelemetry-compatible types."""
    if value is None:
        return None
    if isinstance(value, (bool, str, bytes, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [str(_sanitize_metadata_value(v)) for v in value]
    # Convert other types to strings
    return str(value)


def get_env_bool(key: str, default: bool) -> bool:
    """从环境变量读取布尔值"""
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes")
