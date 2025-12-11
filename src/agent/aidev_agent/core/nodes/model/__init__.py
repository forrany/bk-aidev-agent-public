# -*- coding: utf-8 -*-
"""Model node package.

This package groups the model node implementation (`build_model_node`) and its
strongly-coupled context processing utilities (previously under
`aidev_agent.core.nodes.context_processor`).
"""

from .basic_middleware import create_structured_chat_prompt_template, create_tool_call_prompt_template, get_beijing_now
from .context_processor import ContextProcessor
from .node import ModelState, build_model_node
from .pydantic_models import ModelNodeSettings
from .token_compression import (
    ChatHistoryCompressionMiddleware,
    CompressionState,
    KnowledgeCompressionMiddleware,
    TokenCompressionMiddleware,
    ToolOutputCompressionMiddleware,
)

__all__ = [
    "ChatHistoryCompressionMiddleware",
    "CompressionState",
    "ContextProcessor",
    "KnowledgeCompressionMiddleware",
    "ModelState",
    "ModelNodeSettings",
    "TokenCompressionMiddleware",
    "ToolOutputCompressionMiddleware",
    "build_model_node",
    "create_tool_call_prompt_template",
    "create_structured_chat_prompt_template",
    "get_beijing_now",
]
