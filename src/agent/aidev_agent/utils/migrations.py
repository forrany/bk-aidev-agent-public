from __future__ import annotations

from typing import Any, Mapping

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from aidev_agent.packages.langchain_core.models.llm_gateway import ChatModel
from aidev_agent.pydantic_models import AgentOptions, KnowledgeSettings, ModelContextSettings

_INDEX_SPECIFIC_SEARCH_KEYS = (
    "with_index_specific_search_init",
    "with_index_specific_search_translation",
    "with_index_specific_search_keywords",
)


def migration_chat_model_non_thinking_from_non_thinking_llm_v1(
    non_thinking_llm: str | None,
) -> BaseChatModel | None:
    """从旧版 ``non_thinking_llm`` 字符串迁移非思考模型实例。"""
    if non_thinking_llm:
        return ChatModel.get_setup_instance(model=non_thinking_llm)
    return None


def _dump_compat_options_part(value: Any) -> dict[str, Any]:
    """导出旧兼容模型中的显式输入值，避免默认值污染迁移结果。"""
    if value is None:
        return {}
    if isinstance(value, BaseModel):
        return {k: v for k, v in value.model_dump(exclude_unset=True).items() if v is not None}
    if isinstance(value, Mapping):
        return {k: v for k, v in value.items() if v is not None}
    return {}


def _split_agent_options(
    agent_options: AgentOptions | Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if agent_options is None:
        return {}, {}

    if isinstance(agent_options, AgentOptions):
        intent_options = agent_options.intent_recognition_options
        knowledge_options = agent_options.knowledge_query_options
    elif isinstance(agent_options, Mapping):
        intent_options = agent_options.get("intent_recognition_options") or agent_options.get("intent_recognition")
        knowledge_options = agent_options.get("knowledge_query_options") or agent_options.get("knowledgebase_settings")
    else:
        intent_options = getattr(agent_options, "intent_recognition_options", None)
        knowledge_options = getattr(agent_options, "knowledge_query_options", None)

    return _dump_compat_options_part(intent_options), _dump_compat_options_part(knowledge_options)


def migration_model_context_options_from_agent_options_v1(
    agent_options: AgentOptions | Mapping[str, Any] | None,
) -> ModelContextSettings:
    """从旧版 ``AgentOptions`` 迁移模型上下文配置。"""
    intent_options, knowledge_options = _split_agent_options(agent_options)
    data: dict[str, Any] = {}

    for key in ("llm_token_limit", "token_limit_margin"):
        if key in knowledge_options:
            data[key] = knowledge_options[key]

    if "tool_output_compress_thrd" in intent_options:
        data["tool_output_compress_thrd"] = intent_options["tool_output_compress_thrd"]
    if "llm_code_agent_type" in intent_options:
        data["llm_code_agent_type"] = intent_options["llm_code_agent_type"]
    elif "agent_type" in intent_options:
        data["llm_code_agent_type"] = intent_options["agent_type"]

    return ModelContextSettings.model_validate(data)


def migration_knowledge_query_options_from_agent_options_v1(
    agent_options: AgentOptions | Mapping[str, Any] | None,
) -> KnowledgeSettings:
    """从旧版 ``AgentOptions`` 迁移知识库检索配置。"""
    intent_options, knowledge_options = _split_agent_options(agent_options)
    data = dict(knowledge_options)

    # 平台字段为 document_fragment_count，aidev_agent 运行时实际读取 knowledge_resource_rough_recall_topk。
    # 旧版同时出现多个字段时，正数平台字段 document_fragment_count 优先，修复历史未转换导致配置不生效的问题。
    if "document_fragment_count" in data and data["document_fragment_count"] > 0:
        data["knowledge_resource_rough_recall_topk"] = data.pop("document_fragment_count")
        data.pop("topk", None)
    elif "knowledge_resource_rough_recall_topk" not in data and "topk" in data:
        data["knowledge_resource_rough_recall_topk"] = data.pop("topk")
    else:
        data.pop("topk", None)

    if data.get("rejection_message") == "":
        data.pop("rejection_message")

    for key in _INDEX_SPECIFIC_SEARCH_KEYS:
        if key in intent_options:
            data[key] = intent_options[key]

    return KnowledgeSettings.model_validate(data)
