from typing import List, Tuple

from langchain_core.callbacks import dispatch_custom_event
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from aidev_agent.core.ag_ui.types import CustomMessageType
from aidev_agent.packages.langchain_core.models.llm_gateway import ChatModel
from aidev_agent.packages.langchain_core.models.utils import is_deepseek_r1_series_models, remove_thinking_process
from aidev_agent.packages.model_management.registry import RegistryPluginMixIn
from aidev_agent.services.pydantic_models import AgentOptions
from aidev_agent.utils.decorator import timeit

HUNYUAN_SPECIFIC_RESPONSE = "很抱歉，我还未学习到如何回答这个问题的内容，暂时无法提供相关信息。"
reg = RegistryPluginMixIn()


def is_structured_data(doc):
    structured_data_file_types = ["csv", "xlsx"]
    if isinstance(doc, Document):
        if not hasattr(doc, "metadata"):
            raise RuntimeError(f"召回的文档没有metadata属性！\n文档格式为 Document\n文档内容为：{doc}\n")
        return "file_type" in doc.metadata and doc.metadata["file_type"] in structured_data_file_types
    elif isinstance(doc, dict):
        if "metadata" not in doc:
            raise RuntimeError(f"召回的文档没有metadata属性！\n文档格式为 dict\n文档内容为：{doc}\n")
        return "file_type" in doc["metadata"] and doc["metadata"]["file_type"] in structured_data_file_types
    else:
        raise RuntimeError(f"不支持的文档格式！\n文档内容为：{doc}\n")


def deduplicate_knowledge_chunks(knowledge_chunks):
    return list({item["metadata"]["uid"]: item for item in knowledge_chunks}.values())


def deduplicate_knowledge_file_paths(knowledge_chunks):
    """按照 file path 进行去重，且只保留 metadata，且按照 fine grained score 进行降序排序"""
    unique_items = list(
        {item["metadata"]["file_path"]: {"metadata": item["metadata"]} for item in knowledge_chunks}.values()
    )
    return sorted(unique_items, key=lambda x: x["metadata"]["fine_grained_score"], reverse=True)


def filter_and_select_topk(items, score_threshold, topk):
    if score_threshold:
        filtered_items = [
            item for item in items if item.get("metadata", {}).get("fine_grained_score", 0) >= score_threshold
        ]
    else:
        filtered_items = items
    sorted_items = sorted(filtered_items, key=lambda x: x["metadata"]["fine_grained_score"], reverse=True)
    return sorted_items[:topk]


def invoke_decorator(agent_options: AgentOptions, invoke_func, llm):
    def wrapper(*args, **kwargs):
        nonlocal llm
        # 根据 https://huggingface.co/deepseek-ai/DeepSeek-R1#usage-recommendations 的建议：
        # Avoid adding a system prompt; all instructions should be contained within the user prompt.
        # NOTE: 目前假设只有第 1 个 message 才可能是 SystemMessage
        if global_llm_model_name := agent_options.intent_recognition_options.non_thinking_llm:
            global_llm = ChatModel.get_setup_instance(
                model=global_llm_model_name,
                streaming=True,
            )
            invoke_func_to_use = global_llm.invoke
            llm = global_llm
        else:
            invoke_func_to_use = invoke_func

        if (
            is_deepseek_r1_series_models(llm)
            and isinstance(args[0][0], SystemMessage)
            and isinstance(args[0][-1], HumanMessage)
        ):
            args[0][-1] = HumanMessage(content=f"{args[0][0].content}\n\n{args[0][-1].content}")
            del args[0][0]

        result = invoke_func_to_use(*args)
        if kwargs.get("llm_input_output"):
            kwargs["llm_input_output"][llm.model_name]["input"].append(args[0])
            kwargs["llm_input_output"][llm.model_name]["output"].append(result.content)
        if is_deepseek_r1_series_models(llm):
            # deepseek-r1 系列模型会有 think 过程，在使用结果的时候需要去除
            result.content = remove_thinking_process(result.content)
            result.content = result.content.strip()

        return result

    return wrapper


@timeit(message="相似度计算小模型")
def calculate_similarity(
    text_pairs: List[Tuple[str, str]],
    similarity_model_gpu_cls: str = "model.self_host.similarity_model.SimilarityModel",
) -> List[float]:
    similarity_model_gpu = reg.get_registered_object(service_name=similarity_model_gpu_cls)
    if text_pairs:
        return similarity_model_gpu.compute_similarity(text_pairs)
    else:
        return []


def dispatch_rag_event_chunk(message: str):
    """Dispatch rag event chunk

    Args:
        message (str): The message to dispatch
        config (RunnableConfig): The runnable configuration
    """
    if not message.endswith("\n"):
        message += "\n"
    dispatch_custom_event(
        CustomMessageType.KNOWLEDGE_RAG_TEXT_CONTENT.value,
        data={"chunk": {"content": message}},
    )
