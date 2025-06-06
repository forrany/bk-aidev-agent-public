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

import concurrent.futures
import copy
import logging
import os
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, ClassVar, Dict, List, Set, Tuple

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import _set_config_context
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aidev_agent.api.bk_aidev import BKAidevApi
from aidev_agent.core.extend.intent.prompts import DEFAULT_INTENT_RECOGNITION_PROMPT_TEMPLATES
from aidev_agent.core.extend.intent.similarity_model import calculate_similarity
from aidev_agent.core.extend.intent.utils import (
    HUNYUAN_SPECIFIC_RESPONSE,
    conditional_dispatch_custom_event,
    deduplicate_knowledge_chunks,
    invoke_decorator,
    is_structured_data,
    retry,
    timeit,
)
from aidev_agent.utils.module_loading import import_string

logger = logging.getLogger(__name__)

intent_recognition_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.getenv("INTENT_RECOGNITION_EXECUTOR_MAX_WORKERS", "10"))
)


class IntentStatus(Enum):
    QA_WITH_RETRIEVED_KNOWLEDGE_RESOURCES = "QA_WITH_RETRIEVED_KNOWLEDGE_RESOURCES"
    AGENT_FINISH_WITH_RESPONSE = "AGENT_FINISH_WITH_RESPONSE"
    DIRECTLY_RESPOND_BY_AGENT = "DIRECTLY_RESPOND_BY_AGENT"
    PROCESS_BY_AGENT = "PROCESS_BY_AGENT"


class Decision(Enum):
    GENERAL_QA = "GENERAL_QA"
    PRIVATE_QA = "PRIVATE_QA"
    QUERY_CLARIFICATION = "QUERY_CLARIFICATION"


class FineGrainedScoreType(Enum):
    LLM = "LLM"
    EXCLUSIVE_SIMILARITY_MODEL = "EXCLUSIVE_SIMILARITY_MODEL"
    EMBEDDING = "EMBEDDING"


class IndependentQueryMode(Enum):
    INIT = "INIT"  # 直接使用原始的，不进行重写
    REWRITE = "REWRITE"  # 根据 chat history 对 query 进行重写
    SUM_AND_CONCATE = "SUM_AND_CONCATE"  # 对 chat history 进行总结，并跟 query 进行拼接


class IntentRecognition(BaseModel):
    intent_recognition_prompt_templates: ClassVar[Dict[str, Any]] = DEFAULT_INTENT_RECOGNITION_PROMPT_TEMPLATES
    template_one: ClassVar[Set[str]] = set(["你好", "谢谢"])

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        parent_factory = getattr(super(cls, cls), "intent_recognition_prompt_templates", {})
        current_factory = getattr(cls, "intent_recognition_prompt_templates", {})
        cls.intent_recognition_prompt_templates = {**parent_factory, **current_factory}

    @property
    def _query_instance(self) -> Callable:
        try:
            obj = import_string("aidev.resource.knowledge_base.services.KnowledgeQueryService")
            return obj.internal_query
        except ImportError:
            # 不能resource则使用sdk
            client = BKAidevApi.get_client()
            return client.knowledge_query

    def _search_knowledge_by_client(self, data: dict):
        if "knowledge_template_id" in data and data["knowledge_template_id"] is None:
            data.pop("knowledge_template_id")
        try:
            logger.info(f"查询知识库： {data}")
            result = self._query_instance(data)
            docs = result["documents"]
            for doc in docs:
                if isinstance(doc, Document):
                    if not hasattr(doc, "metadata"):
                        raise RuntimeError(f"召回的文档缺少 metadata 字段！\n文档内容为：{doc}\n")
                    if "__score__" not in doc.metadata:
                        raise RuntimeError(f"召回的文档缺少 __score__ 字段！\n文档内容为：{doc}\n")
                elif isinstance(doc, dict):
                    if "metadata" not in doc:
                        raise RuntimeError(f"召回的文档缺少 metadata 字段！\n文档内容为：{doc}\n")
                    if "__score__" not in doc["metadata"]:
                        raise RuntimeError(f"召回的文档缺少 __score__ 字段！\n文档内容为：{doc}\n")
                else:
                    raise RuntimeError(f"召回文档格式有误！\n文档内容为：{doc}\n")
            return docs
        except Exception as err:
            logger.error(f"\n\n=====\n>>>>> 知识库查询接口调用出错！\n\ndata 内容为：\n{data}\n\n error: {err}")
            raise

    def _construct_index_query_kwargs(
        self, index_query_kwargs, query, knowledges, knowledge_type, resource_type, **kwargs
    ):
        knowledge_type_to_id_type = {
            "knowledge_items": "knowledge_id",
            "knowledge_bases": "knowledge_base_id",
        }
        if resource_type == "knowledge":
            custom_index_name_key = "knowledge_resource_index_names"
        elif resource_type == "tool":
            custom_index_name_key = "tool_resource_index_names"
        else:
            raise ValueError(f"不支持的 resource 类型：{resource_type}")
        if knowledges:
            supported_ids = [knowledge.get("id") for knowledge in knowledges]
            for knowledge in knowledges:
                all_index_names = []
                supported_index_names = []
                if index_config := knowledge.get("index_config"):
                    for index_type in ["full_text_indexes", "vector_indexes"]:
                        if indexes := index_config.get(index_type):
                            for index in indexes:
                                if index_name := index.get("index_name"):
                                    supported_index_names.append(index_name)

                    custom_index_names = kwargs.get(custom_index_name_key, {})
                    custom_index_names_type = custom_index_names.get(knowledge_type)
                    if custom_index_names and custom_index_names_type:
                        if not set(list(custom_index_names_type.keys())).issubset(set(supported_ids)):
                            raise ValueError(
                                f"传入的 {knowledge_type} 类型的 ID 有：{supported_ids}，"
                                f"但传入的 {knowledge_type} 类型的自定义的向量索引 ID 有："
                                f"{list(custom_index_names_type.keys())}，"
                                "请确保后者是前者的子集！"
                            )
                        if custom_index_names_type_id := custom_index_names_type.get(knowledge.get("id")):
                            if not set(custom_index_names_type_id).issubset(set(supported_index_names)):
                                raise ValueError(
                                    f"{knowledge_type} 类型的知识（库）ID {knowledge.get('id')} "
                                    f"支持的向量索引有：{supported_index_names}，"
                                    f"但传入的自定义向量索引为：{custom_index_names_type_id}，"
                                    "请传入支持的向量索引的子集！"
                                )
                            all_index_names = custom_index_names_type_id
                if not all_index_names:
                    all_index_names = supported_index_names
                if not all_index_names:
                    raise RuntimeError(f"{knowledge_type} 类型的知识（库）ID {knowledge.get('id')} 的索引为空！")
                index_query_kwargs.extend(
                    [
                        {
                            "index_name": index_name,
                            "index_value": query,
                            knowledge_type_to_id_type[knowledge_type]: knowledge["id"],
                        }
                        for index_name in all_index_names
                    ]
                )

    @timeit(message="知识库检索（index_specific方式）")
    def search_knowledge_index_specific(
        self, knowledge_items: list[dict], knowledge_bases: list[dict], query, topk, resource_type="knowledge", **kwargs
    ):
        """基于向量检索获取相关文档（index_specific方式）"""
        index_query_kwargs = []
        self._construct_index_query_kwargs(
            index_query_kwargs, query, knowledge_items, "knowledge_items", resource_type=resource_type, **kwargs
        )
        self._construct_index_query_kwargs(
            index_query_kwargs, query, knowledge_bases, "knowledge_bases", resource_type=resource_type, **kwargs
        )
        data = {
            "query": query,
            "topk": topk,
            "index_query_kwargs": index_query_kwargs,
            "knowledge_template_id": kwargs.get("knowledge_template_id"),
            "with_scalar_data": kwargs.get("with_scalar_data", True),
            "raw": True,  # 知识库查询接口集成了本文件中的重排逻辑，设置为True防止循环重排。下同
            "type": "index_specific",
        }
        return self._search_knowledge_by_client(data)

    @timeit(message="知识库检索（index_specific方式，使用提取的关键词）")
    def search_knowledge_index_specific_keywords(
        self, knowledge_items: list[dict], knowledge_bases: list[dict], extracted_keywords, topk, **kwargs
    ):
        """基于index_specific获取相关文档（index_specific方式，使用提取的关键词）"""
        if extracted_keywords:
            return self.search_knowledge_index_specific(
                knowledge_items=knowledge_items,
                knowledge_bases=knowledge_bases,
                query="\n\n".join(extracted_keywords),
                topk=topk,
                disable_timeit=True,
                **kwargs,
            )
        else:
            return []

    @timeit(message="知识库检索（index_specific方式，使用翻译后的中文）")
    def search_knowledge_index_specific_translation(
        self, knowledge_items: list[dict], knowledge_bases: list[dict], translated_query, topk, **kwargs
    ):
        """基于index_specific获取相关文档（index_specific方式，使用翻译后的中文）"""
        if translated_query:
            return self.search_knowledge_index_specific(
                knowledge_items=knowledge_items,
                knowledge_bases=knowledge_bases,
                query=translated_query,
                topk=topk,
                disable_timeit=True,
                **kwargs,
            )
        else:
            return []

    @timeit(message="知识库检索（nature方式）")
    def search_knowledge_nature(self, knowledge_items: list[dict], knowledge_bases: list[dict], query, topk, **kwargs):
        """基于向量检索获取相关文档（nature方式）"""
        data = {
            "query": query,
            "topk": topk,
            "knowledge_id": [knowledge["id"] for knowledge in knowledge_items],
            "knowledge_base_id": [knowledge["id"] for knowledge in knowledge_bases],
            "knowledge_template_id": kwargs.get("knowledge_template_id"),
            "with_scalar_data": kwargs.get("with_scalar_data", True),
            "raw": True,
            "type": "nature",
        }
        return self._search_knowledge_by_client(data)

    def _es_client(self):
        raise NotImplementedError

    def _parse_es_hits(self, es_resp):
        raise NotImplementedError

    @timeit(message="知识库检索（ES方式，使用完整query）")
    def search_knowledge_es_query(
        self, knowledge_items: list[dict], knowledge_bases: list[dict], query, topk, **kwargs
    ):
        """基于ES获取相关文档（ES方式，使用完整query）"""
        raise NotImplementedError

    @timeit(message="知识库检索（ES方式，使用提取的关键词）")
    def search_knowledge_es_keywords(
        self, knowledge_items: list[dict], knowledge_bases: list[dict], extracted_keywords, topk, **kwargs
    ):
        raise NotImplementedError

    @timeit(message="用户提问关键词提取")
    @retry(max_retries=5, max_seconds=3600)
    def extract_query_keywords(self, query, llm, **kwargs):
        sys_prompt = self.__class__.intent_recognition_prompt_templates.get(
            "extract_query_keywords_sys_prompt_template"
        )
        usr_prompt = self.__class__.intent_recognition_prompt_templates.get(
            "extract_query_keywords_usr_prompt_template"
        ).render(query=query)
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=usr_prompt),
        ]
        # TODO: 待确认：并发请求内部无法 dispatch_custom_event，所以无需调用 conditional_dispatch_custom_event
        invoke_func = invoke_decorator(llm.invoke, llm)
        resp = invoke_func(messages)
        resp_content = resp.content
        extracted_keywords = resp_content.strip().split("\n")
        extracted_keywords = list(filter(None, extracted_keywords))
        logger.info(f"=====> <extract_query_keywords的结果>：{extracted_keywords}")
        return extracted_keywords

    @timeit(message="用户提问翻译")
    @retry(max_retries=5, max_seconds=3600)
    def query_translation(self, query, llm, **kwargs):
        sys_prompt = self.__class__.intent_recognition_prompt_templates.get("query_translation_sys_prompt_template")
        usr_prompt = self.__class__.intent_recognition_prompt_templates.get(
            "query_translation_usr_prompt_template"
        ).render(query=query)
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=usr_prompt),
        ]
        # TODO: 待确认：并发请求内部无法 dispatch_custom_event，所以无需调用 conditional_dispatch_custom_event
        invoke_func = invoke_decorator(llm.invoke, llm)
        resp = invoke_func(messages)
        resp_content = resp.content
        logger.info(f"=====> <query_translation的结果>：{resp_content}")
        if resp_content.strip() == "None":
            return None
        else:
            return resp_content

    @timeit(message="意图切换检测")
    @retry(max_retries=5, max_seconds=3600)
    def latest_query_classification(self, chat_history, query, llm, **kwargs):
        sys_prompt = self.__class__.intent_recognition_prompt_templates.get(
            "latest_query_classification_sys_prompt_template"
        )
        usr_prompt = self.__class__.intent_recognition_prompt_templates.get(
            "latest_query_classification_usr_prompt_template"
        ).render(chat_history=chat_history, query=query)
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=usr_prompt),
        ]
        conditional_dispatch_custom_event("custom_event", {"front_end_display": False}, **kwargs)
        invoke_func = invoke_decorator(llm.invoke, llm)
        resp = invoke_func(messages, **kwargs)
        conditional_dispatch_custom_event("custom_event", {"front_end_display": True}, **kwargs)
        resp_content = resp.content
        logger.info(f"=====> <query分类结果>：{resp_content}")
        if "<<<<<new>>>>>" in resp_content:
            return "new"
        elif "<<<<<continue>>>>>" in resp_content:
            return "continue"
        elif "<<<<<finish>>>>>" in resp_content:
            return "finish"
        else:
            return "new"  # 其余所有边缘情况默认直接重新开始

    @timeit(message="独立查询重写")
    @retry(max_retries=5, max_seconds=3600)
    def query_rewrite_for_independence(self, chat_history, query, llm, display=False, **kwargs):
        """
        :param display: 是否将独立查询重写的结果也展示在前端
        """
        sys_prompt = self.__class__.intent_recognition_prompt_templates.get(
            "query_rewrite_for_independence_sys_prompt_template"
        )
        usr_prompt = self.__class__.intent_recognition_prompt_templates.get(
            "query_rewrite_for_independence_usr_prompt_template"
        ).render(chat_history=chat_history, query=query)
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=usr_prompt),
        ]
        if display:
            conditional_dispatch_custom_event(
                "custom_event",
                {"custom_return_chunk": "结合历史对话信息，您似乎是想问：“"},
                **kwargs,
            )
            invoke_func = invoke_decorator(llm.invoke, llm)
            resp = invoke_func(messages, **kwargs)
            conditional_dispatch_custom_event(
                "custom_event",
                {"custom_return_chunk": "”。接下来我尝试进行回答。\n\n"},
                **kwargs,
            )
        else:
            # 包在这 2 行 conditional_dispatch_custom_event 代码之间的 LLM 输出不会在前端展示
            conditional_dispatch_custom_event("custom_event", {"front_end_display": False}, **kwargs)
            invoke_func = invoke_decorator(llm.invoke, llm)
            resp = invoke_func(messages, **kwargs)
            conditional_dispatch_custom_event("custom_event", {"front_end_display": True}, **kwargs)
        resp_content = resp.content
        logger.info(f"=====> <query重写结果>：{resp_content}")
        return resp_content

    @timeit(message="意图切换检测和独立查询重写/直接答复")
    @retry(max_retries=5, max_seconds=3600)
    def query_cls_with_resp_or_rewrite(self, chat_history, query, llm, **kwargs):
        sys_prompt = self.__class__.intent_recognition_prompt_templates.get(
            "query_cls_with_resp_or_rewrite_sys_prompt_template"
        )
        usr_prompt = self.__class__.intent_recognition_prompt_templates.get(
            "query_cls_with_resp_or_rewrite_usr_prompt_template"
        ).render(chat_history=chat_history, query=query)
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=usr_prompt),
        ]
        conditional_dispatch_custom_event("custom_event", {"front_end_display": False}, **kwargs)
        invoke_func = invoke_decorator(llm.invoke, llm)
        resp = invoke_func(messages)
        conditional_dispatch_custom_event("custom_event", {"front_end_display": True}, **kwargs)
        resp_content = resp.content
        logger.info(f"=====> <query_cls_with_resp_or_rewrite的结果>：{resp_content}")
        if resp_content.startswith("<<<<<new>>>>>"):
            return {
                "query_cls": "new",
            }
        elif resp_content.startswith("<<<<<continue>>>>>"):
            if resp_content.startswith("<<<<<continue>>>>>$REWRITTEN_QUERY: "):
                rewritten_query = resp_content[len("<<<<<continue>>>>>$REWRITTEN_QUERY: ") :]
            else:
                rewritten_query = resp_content[len("<<<<<continue>>>>>") :]
            return {
                "query_cls": "continue",
                "rewritten_query": rewritten_query,
            }
        elif resp_content.startswith("<<<<<finish>>>>>"):
            if resp_content.startswith("<<<<<finish>>>>>$RESPONSE: "):
                response = resp_content[len("<<<<<finish>>>>>$RESPONSE: ") :]
            else:
                response = resp_content[len("<<<<<finish>>>>>") :]
            return {
                "query_cls": "finish",
                "response": response,  # TODO: 后半段stream化
            }
        else:
            return {
                "query_cls": "new",
            }  # 其余所有边缘情况默认直接重新开始

    @timeit(message="伪工具类资源描述生成")
    @retry(max_retries=5, max_seconds=3600)
    def gen_pseudo_tool_resource_description(self, query, llm, **kwargs):
        sys_prompt = self.__class__.intent_recognition_prompt_templates.get(
            "gen_pseudo_tool_resource_description_sys_prompt_template"
        )
        usr_prompt = self.__class__.intent_recognition_prompt_templates.get(
            "gen_pseudo_tool_resource_description_usr_prompt_template"
        ).render(query=query)
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=usr_prompt),
        ]
        conditional_dispatch_custom_event("custom_event", {"front_end_display": False}, **kwargs)
        invoke_func = invoke_decorator(llm.invoke, llm)
        resp = invoke_func(messages)
        conditional_dispatch_custom_event("custom_event", {"front_end_display": True}, **kwargs)
        resp_content = resp.content
        logger.info(f"=====> <伪资源描述重写结果>：{resp_content}")
        return resp_content

    def intent_recognition_by_code(self, intent_code):
        """
        根据特殊编码判断意图
        如果命中，则返回对应意图
        如果没命中，则返回 None
        """
        return None

    def intent_recognition_by_template(self, query):
        """
        根据意图关键词/模板匹配判断意图
        如果命中，则返回对应意图
        如果没命中，则返回 None
        """
        return None

    def intent_recognition_by_template_one(self, query):
        """
        如果命中提问列表，则直接快速单跳答复
        如果没命中，则返回 None
        """
        if query in self.__class__.template_one:
            return "directly_respond"
        else:
            return None

    def intent_recognition_by_exclusive_model(self, query):
        """
        根据定制的意图识别小模型判断意图
        如果命中，则返回对应意图
        如果没命中，则返回 None
        """
        return None

    def weighted_reciprocal_rank_fusion(self, searched_docs, weights, k=60):
        """
        加权倒数排名融合算法。

        参数:
        :param results: 一个列表，其中每个元素是一个包含文档的列表。
        :param weights: 一个列表，表示每个检索支路的权重。
        :param k: 一个整数，表示排名的最大考虑深度。

        返回:
        - 一个按照融合分数降序排序后的文档列表，每个文档的 `metadata` 字段中增加一个 `rrf_score` 字段。
        """

        if len(searched_docs) != len(weights):
            raise ValueError("结果列表和权重列表的长度必须相同。")

        fusion_scores = defaultdict(float)
        doc_content = {}

        for result, weight in zip(searched_docs, weights):
            for rank, doc in enumerate(result):
                if rank < k:
                    doc_id = doc["metadata"]["uid"]
                    fusion_scores[doc_id] += weight / (rank + 1)
                    if doc_id not in doc_content:
                        doc_content[doc_id] = doc

        for doc_id, score in fusion_scores.items():
            doc_content[doc_id]["metadata"]["rrf_score"] = score

        sorted_docs = sorted(doc_content.values(), key=lambda x: x["metadata"]["rrf_score"], reverse=True)

        return sorted_docs

    def calculate_fine_grained_scores(
        self,
        fine_grained_score_type,
        query_for_search,
        llm,
        context_docs_with_scores,
        **kwargs,
    ):
        if fine_grained_score_type == FineGrainedScoreType.LLM:
            # NOTE: 如果 FineGrainedScoreType 为 LLM，则因为当前只有是/否相关的判断，因此分数只有 1.0 或 0.0
            fine_grained_scores = self.llm_relevance_determiner_parallel(
                (
                    kwargs.get("translated_query", query_for_search)
                    if kwargs.get("use_independent_query_in_scores", True)
                    else kwargs["input"]
                ),
                [doc for doc, _ in context_docs_with_scores],
                llm,
                **kwargs,
            )
        elif fine_grained_score_type == FineGrainedScoreType.EXCLUSIVE_SIMILARITY_MODEL:
            # 使用专属小模型计算的分数作为最终的细粒度相似度分数
            # TODO: 目前知识类资源和工具类资源是独立使用小模型的，可以考虑在都要计算的情况下，合成一个batch进行计算
            # NOTE: 如果有 index_content 且是结构化数据则取 index_content，否则才取 page_content（兼容写法）。
            # 待知识库后台对非结构化数据的处理方式的 index_content 不是默认使用LLM总结后的内容之后，
            # 可将“且是结构化数据”的逻辑去除。
            # NOTE: 目前暂不考虑检索返回模板对 page_content 的影响
            fine_grained_scores = calculate_similarity(
                [
                    (
                        (
                            kwargs.get("translated_query", query_for_search)
                            if kwargs.get("use_independent_query_in_scores", False)
                            else kwargs["input"]
                        ),
                        (
                            doc.metadata["index_content"]
                            if "index_content" in doc.metadata and is_structured_data(doc)
                            else doc.page_content
                        ),
                    )
                    for doc, _ in context_docs_with_scores
                ]
            )
            fine_grained_scores = [float(fine_grained_score) for fine_grained_score in fine_grained_scores]
        elif fine_grained_score_type == FineGrainedScoreType.EMBEDDING:
            # 直接使用emb分数作为最终的细粒度相似度分数
            fine_grained_scores = [float(emb_score) for _, emb_score in context_docs_with_scores]
        else:
            raise ValueError(
                f"当前仅支持以下计算细粒度相关分数的方式：{[score_type.value for score_type in FineGrainedScoreType]}，"
                f"但传入的 fine_grained_score_type 为：`{fine_grained_score_type}`"
            )

        return fine_grained_scores

    def separate_docs_by_scores(self, context_docs_with_scores, fine_grained_scores, reject_threshold):
        contexts_emb_recalled = []
        contexts_lowly_relevant = []
        contexts_moderately_relevant = []
        contexts_highly_relevant = []
        for context_doc_with_score, fine_grained_score in zip(context_docs_with_scores, fine_grained_scores):
            context_doc_with_fine_grained_score = copy.deepcopy(context_doc_with_score[0].dict())
            context_doc_with_fine_grained_score["metadata"]["fine_grained_score"] = fine_grained_score
            contexts_emb_recalled.append(context_doc_with_fine_grained_score)
            if fine_grained_score < reject_threshold[0]:
                contexts_lowly_relevant.append(context_doc_with_fine_grained_score)
            elif reject_threshold[0] <= fine_grained_score < reject_threshold[1]:
                contexts_moderately_relevant.append(context_doc_with_fine_grained_score)
            elif fine_grained_score >= reject_threshold[1]:
                contexts_highly_relevant.append(context_doc_with_fine_grained_score)

        return (
            contexts_emb_recalled,
            contexts_lowly_relevant,
            contexts_moderately_relevant,
            contexts_highly_relevant,
        )

    @retry(max_retries=5, max_seconds=3600)
    def llm_relevance_determiner(self, query, doc, llm, **kwargs):
        # NOTE: 如果有 index_content 且是结构化数据则取 index_content，否则才取 page_content（兼容写法）。
        # 待知识库后台对非结构化数据的处理方式的 index_content 不是默认使用LLM总结后的内容之后，
        # 可将“且是结构化数据”的逻辑去除。
        # NOTE: 目前暂不考虑检索返回模板对 page_content 的影响
        if len(query) > len(kwargs["input"]) and query.endswith(f"\n{kwargs['input']}"):  # 拼接场景
            his_sum = query[: -(len(kwargs["input"]) + 1)]
            sys_prompt = self.__class__.intent_recognition_prompt_templates.get(
                "llm_relevance_determiner_concate_sys_prompt_template"
            )
            usr_prompt = self.__class__.intent_recognition_prompt_templates.get(
                "llm_relevance_determiner_concate_usr_prompt_template"
            ).render(
                his_sum=his_sum,
                query=kwargs["input"],
                doc=(
                    doc.metadata["index_content"]
                    if "index_content" in doc.metadata and is_structured_data(doc)
                    else doc.page_content
                ),
            )
        else:
            sys_prompt = self.__class__.intent_recognition_prompt_templates.get(
                "llm_relevance_determiner_sys_prompt_template"
            )
            usr_prompt = self.__class__.intent_recognition_prompt_templates.get(
                "llm_relevance_determiner_usr_prompt_template"
            ).render(
                query=query,
                doc=(
                    doc.metadata["index_content"]
                    if "index_content" in doc.metadata and is_structured_data(doc)
                    else doc.page_content
                ),
            )
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=usr_prompt),
        ]
        # TODO: 待确认：并发请求内部无法 dispatch_custom_event，所以无需调用 conditional_dispatch_custom_event
        invoke_func = invoke_decorator(llm.invoke, llm)
        resp = invoke_func(messages)
        resp_content = resp.content
        return not resp_content.startswith("0")  # 用0来判断，减少误删

    @timeit(message="使用LLM并发进行query和召回文档相关性判断")
    def llm_relevance_determiner_parallel(self, query, fusion_docs, llm, **kwargs):
        try:
            futures = [
                intent_recognition_executor.submit(self.llm_relevance_determiner, query, doc, llm, **kwargs)
                for doc in fusion_docs
            ]
            results = [1.0 if future.result() else 0.0 for future in futures]
        except Exception:
            # 如果 LLM 调用失败则不进行过滤
            results = [1.0] * len(fusion_docs)
            logger.warning("调用 LLM 来判断提问和知识相关性时失败，因此不进行过滤！")
        logger.info(f"=====> < llm_relevance_determiner_parallel 的结果>：{results}")
        return results

    @retry(max_retries=5, max_seconds=3600)
    def llm_context_compressor(self, provided_chat_history, query, candidate_context, llm, **kwargs):
        # 默认使用 specific 方式。
        compressor_type = kwargs.get("llm_context_compressor_type", "specific")
        if compressor_type == "common":
            sys_prompt = self.__class__.intent_recognition_prompt_templates.get(
                "llm_common_compressor_sys_prompt_template"
            )
            usr_prompt = self.__class__.intent_recognition_prompt_templates.get(
                "llm_common_compressor_usr_prompt_template"
            ).render(content=candidate_context)
        elif compressor_type == "specific":
            sys_prompt = self.__class__.intent_recognition_prompt_templates.get(
                "llm_context_compressor_sys_prompt_template"
            )
            usr_prompt = self.__class__.intent_recognition_prompt_templates.get(
                "llm_context_compressor_usr_prompt_template"
            ).render(
                provided_chat_history=provided_chat_history,
                query=query,
                candidate_context=candidate_context,
            )
        else:
            raise ValueError(f"不支持的知识库知识压缩方式：{compressor_type}")
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=usr_prompt),
        ]
        # TODO: 待确认：并发请求内部无法 dispatch_custom_event，所以无需调用 conditional_dispatch_custom_event
        invoke_func = invoke_decorator(llm.invoke, llm)
        resp = invoke_func(messages)
        resp_content = resp.content
        # 如果触发了混元的特殊回复，则不进行压缩
        if resp_content == HUNYUAN_SPECIFIC_RESPONSE:
            resp_content = candidate_context
        return resp_content

    @timeit(message="使用LLM并发进行知识库内容压缩总结")
    def llm_context_compressor_parallel(self, provided_chat_history, query, context, llm, **kwargs):
        try:
            futures = [
                intent_recognition_executor.submit(
                    self.llm_context_compressor, provided_chat_history, query, candidate_context, llm, **kwargs
                )
                for candidate_context in context
            ]
            results = [future.result() for future in futures]
        except Exception:
            # 如果 LLM 调用失败则不进行总结
            results = context
            logger.warning("调用 LLM 来对知识库内容进行压缩总结时失败，因此不进行总结！")
        return results

    @retry(max_retries=5, max_seconds=3600)
    def llm_intermediate_step_compressor(self, provided_chat_history, query, intermediate_step, llm, **kwargs):
        # 注：如果使用 hunyuan-turbo，使用带query和会话历史的复杂压缩方式效果很差。
        # 例如用户最新提问如下：```今天又有啥大新闻？```"这样的例子，
        # 混元经常直接返回“很抱歉，我还未学习到如何回答这个问题的内容，暂时无法提供相关信息。”
        # 而不是压缩所提供查询工具调用得到的结果。
        # 所以默认使用直接总结的方式：common
        # NOTE 更新：上述话术可能是触发混元的安全检查导致返回该内容。因此默认还是使用 specific 方式。
        compressor_type = kwargs.get("llm_context_compressor_type", "specific")
        if compressor_type == "common":
            sys_prompt = self.__class__.intent_recognition_prompt_templates.get(
                "llm_common_compressor_sys_prompt_template"
            )
            usr_prompt = self.__class__.intent_recognition_prompt_templates.get(
                "llm_common_compressor_usr_prompt_template"
            ).render(content=str(intermediate_step[1]))
        elif compressor_type == "specific":
            sys_prompt = self.__class__.intent_recognition_prompt_templates.get(
                "llm_intermediate_step_compressor_sys_prompt_template"
            ).render(
                candidate_tool_name=intermediate_step[0].tool,
            )
            usr_prompt = self.__class__.intent_recognition_prompt_templates.get(
                "llm_intermediate_step_compressor_usr_prompt_template"
            ).render(
                provided_chat_history=provided_chat_history,
                query=query,
                candidate_tool_name=intermediate_step[0].tool,
                candidate_tool_result=str(intermediate_step[1]),
            )
        else:
            raise ValueError(f"不支持的工具调用结果压缩方式：{compressor_type}")
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=usr_prompt),
        ]
        # TODO: 待确认：并发请求内部无法 dispatch_custom_event，所以无需调用 conditional_dispatch_custom_event
        invoke_func = invoke_decorator(llm.invoke, llm)
        resp = invoke_func(messages)
        resp_content = resp.content
        # 如果触发了混元的特殊回复，则不进行压缩
        if resp_content == HUNYUAN_SPECIFIC_RESPONSE:
            resp_content = str(intermediate_step[1])
        return resp_content

    @timeit(message="使用LLM并发进行工具调用结果压缩总结")
    def llm_intermediate_step_compressor_parallel(
        self, provided_chat_history, query, intermediate_steps, llm, **kwargs
    ):
        futures = {
            intent_recognition_executor.submit(
                self.llm_intermediate_step_compressor, provided_chat_history, query, intermediate_step, llm, **kwargs
            ): idx
            for idx, intermediate_step in enumerate(intermediate_steps)
        }
        results = [None] * len(intermediate_steps)
        try:
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(
                        f"调用 LLM 来对工具调用结果进行压缩总结时，LLM 调用失败，索引 {idx}，错误：{e}。因此该内容不进行总结。"
                    )
        except Exception as e:
            logger.warning(f"调用 LLM 来对工具调用结果进行压缩总结时，LLM 调用失败，错误：{e}。因此不进行总结。")
        try:
            for intermediate_step_idx in range(len(intermediate_steps)):
                if results[intermediate_step_idx] is not None:
                    intermediate_steps[intermediate_step_idx] = (
                        intermediate_steps[intermediate_step_idx][0],
                        results[intermediate_step_idx],
                    )
        except Exception as e:
            logger.warning(
                f"调用 LLM 来对工具调用结果进行压缩总结时，执行结果解析和替换失败，错误：{e}。因此不进行总结。"
            )

    @timeit(message="知识库检索（self query方式，使用完整query）")
    @retry(max_retries=5, max_seconds=3600)
    def search_knowledge_self_query(self, query_for_search, llm, **kwargs):
        # TODO: 这里需要考虑是否将 self-query 模块接口化，然后这里就只通过接口的方式调用即可，方便本地调试
        # TODO: 以下为临时写法，用于串通整个流程
        # TODO: self_query_retriever是有可能执行失败的，需要retry机制，以及最终实在失败了的处理机制
        # NOTE NOTE NOTE NOTE NOTE TODO:
        # 后续步骤LLM判断可回答性的prompt还得针对性优化，或者使用不同的prompt分支，因为对于召回的某一行，
        # 类似“成绩高于85分、18岁以上的学生有多少个”的提问LLM会觉得不可回答，
        # 类似“成绩高于85分、18岁以上的学生有哪些”的提问LLM会觉得可回答。
        # 这类求总数量的query确实会比较特殊，单条知识会导致LLM觉得不可回答，都返回了个0
        raise NotImplementedError

    @timeit(message="知识类资源粗召+精排")
    def retrieve_and_parse_knowledge_resource(
        self,
        query,
        llm,
        with_structured_data,  # 用户勾选的知识中是否包含了结构化数据
        with_index_specific_search,
        with_index_specific_search_init,
        with_index_specific_search_translation,
        with_index_specific_search_keywords,
        with_es_search_query,
        with_es_search_keywords,
        with_rrf,
        knowledge_items,
        knowledge_bases,
        knowledge_resource_rough_recall_topk,
        knowledge_resource_reject_threshold,
        knowledge_resource_fine_grained_score_type,
        self_query_threshold_top_n,
        **kwargs,
    ):
        """知识类资源检索并解析结果"""
        query_for_search = query
        if not any(
            [
                with_index_specific_search,
                with_index_specific_search_init,
                with_index_specific_search_translation,
                with_index_specific_search_keywords,
                with_es_search_query,
                with_es_search_keywords,
            ]
        ):
            raise RuntimeError("请至少选择一种召回方式！")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if with_index_specific_search:
                future_index_specific = executor.submit(
                    self.search_knowledge_index_specific,
                    knowledge_items=knowledge_items,
                    knowledge_bases=knowledge_bases,
                    query=query_for_search,
                    topk=knowledge_resource_rough_recall_topk,
                    **kwargs,
                )
            if with_index_specific_search_init and query_for_search != kwargs["input"]:
                future_index_specific_init = executor.submit(
                    self.search_knowledge_index_specific,
                    knowledge_items=knowledge_items,
                    knowledge_bases=knowledge_bases,
                    query=kwargs["input"],
                    topk=knowledge_resource_rough_recall_topk,
                    **kwargs,
                )
            if with_index_specific_search_translation:
                future_translated_query = executor.submit(
                    self.query_translation,
                    query=(
                        query_for_search
                        if kwargs.get("use_independent_query_in_translation", False)
                        else kwargs["input"]
                    ),
                    llm=llm,
                    **kwargs,
                )
                translated_query = future_translated_query.result()
                future_index_specific_translation = executor.submit(
                    self.search_knowledge_index_specific_translation,
                    knowledge_items=knowledge_items,
                    knowledge_bases=knowledge_bases,
                    translated_query=translated_query,
                    topk=knowledge_resource_rough_recall_topk,
                    **kwargs,
                )
                if kwargs.get("use_translated_query_in_scores", True) and translated_query:
                    kwargs["translated_query"] = translated_query
            if with_es_search_query:
                future_es_query = executor.submit(
                    self.search_knowledge_es_query,
                    knowledge_items=knowledge_items,
                    knowledge_bases=knowledge_bases,
                    query=query_for_search,
                    topk=knowledge_resource_rough_recall_topk,
                    **kwargs,
                )
            if with_index_specific_search_keywords or with_es_search_keywords:
                future_extracted_keywords = executor.submit(
                    self.extract_query_keywords,
                    query=query_for_search,
                    llm=llm,
                    **kwargs,
                )
            if with_index_specific_search_keywords:
                future_index_specific_keywords = executor.submit(
                    self.search_knowledge_index_specific_keywords,
                    knowledge_items=knowledge_items,
                    knowledge_bases=knowledge_bases,
                    extracted_keywords=future_extracted_keywords.result(),
                    topk=knowledge_resource_rough_recall_topk,
                    **kwargs,
                )
            if with_es_search_keywords:
                future_es_keywords = executor.submit(
                    self.search_knowledge_es_keywords,
                    knowledge_items=knowledge_items,
                    knowledge_bases=knowledge_bases,
                    extracted_keywords=future_extracted_keywords.result(),
                    topk=knowledge_resource_rough_recall_topk,
                    **kwargs,
                )
            # TODO: 去除 nature 分支
            if with_structured_data:
                future_nature = executor.submit(
                    self.search_knowledge_nature,
                    knowledge_items=knowledge_items,
                    knowledge_bases=knowledge_bases,
                    query=query_for_search,
                    topk=knowledge_resource_rough_recall_topk,
                    **kwargs,
                )

            retrieved_results_index_specific = future_index_specific.result() if with_index_specific_search else []

            if with_index_specific_search_init and query_for_search != kwargs["input"]:
                retrieved_results_index_specific_init = future_index_specific_init.result()
            else:
                retrieved_results_index_specific_init = []

            if with_index_specific_search_translation:
                retrieved_results_index_specific_translation = future_index_specific_translation.result()
            else:
                retrieved_results_index_specific_translation = []

            if with_index_specific_search_keywords:
                retrieved_results_index_specific_keywords = future_index_specific_keywords.result()
            else:
                retrieved_results_index_specific_keywords = []

            retrieved_results_es_query = future_es_query.result() if with_es_search_query else []

            retrieved_results_es_keywords = future_es_keywords.result() if with_es_search_keywords else []

            retrieved_results_nature = future_nature.result() if with_structured_data else []

        if with_rrf:
            fusion_docs = self.weighted_reciprocal_rank_fusion(
                [
                    retrieved_results_index_specific,
                    retrieved_results_index_specific_init,
                    retrieved_results_index_specific_translation,
                    retrieved_results_index_specific_keywords,
                    retrieved_results_es_query,
                    retrieved_results_es_keywords,
                    retrieved_results_nature,
                ],
                weights=[1.0 / 7] * 7,
            )
        else:
            # NOTE: 推荐使用 with_rrf，否则因为ES支路的score使用该次召回的最大值来归一化的，最高分就是1，
            # 相当于会更照顾ES支路的召回结果
            fusion_docs = sorted(
                deduplicate_knowledge_chunks(
                    retrieved_results_index_specific
                    + retrieved_results_index_specific_init
                    + retrieved_results_index_specific_translation
                    + retrieved_results_index_specific_keywords
                    + retrieved_results_es_query
                    + retrieved_results_es_keywords
                    + retrieved_results_nature
                ),
                key=lambda x: x["metadata"]["__score__"],
                reverse=True,
            )

        # TODO: 这里也可以考虑先使用 rerank 小模型排个序再取 self_query_threshold_top_n 文档来判断 query 是否涉及结构化数据
        # TODO: 待去除 nature 分支后即可正式走以下流程：
        if with_structured_data and any([is_structured_data(doc) for doc in fusion_docs[:self_query_threshold_top_n]]):
            # 在这种情况下需要使用 self-query 模块进行 2 次召回
            re_retrieved_res = self.search_knowledge_self_query(query_for_search, llm, **kwargs)
            fusion_docs.extend(re_retrieved_res)

        context_docs_with_scores = [(Document(**item), item["metadata"]["__score__"]) for item in fusion_docs]
        fine_grained_scores = self.calculate_fine_grained_scores(
            knowledge_resource_fine_grained_score_type,
            query_for_search,
            llm,
            context_docs_with_scores,
            **kwargs,
        )
        return self.separate_docs_by_scores(
            context_docs_with_scores, fine_grained_scores, knowledge_resource_reject_threshold
        )

    @timeit(message="工具类资源粗召+精排")
    def retrieve_and_parse_tool_resource(
        self,
        query,
        llm,
        tool_resource_item_ids,
        tool_resource_base_ids,
        tool_resource_rough_recall_topk,
        tool_resource_reject_threshold,
        tool_resource_fine_grained_score_type,
        gen_pseudo_tool_resource_desc=None,
        **kwargs,
    ):
        """工具类资源检索并解析结果
        TODO: 暂无工具类知识库
        """
        raise NotImplementedError

    def make_decision(
        self,
        knowledge_resources_emb_recalled,
        knowledge_resources_lowly_relevant,
        knowledge_resources_moderately_relevant,
        knowledge_resources_highly_relevant,
        tool_resources_emb_recalled,
        tool_resources_lowly_relevant,
        tool_resources_moderately_relevant,
        tool_resources_highly_relevant,
    ):
        """决策分类

        # NOTE: 目前将粗召（+精排）后高于阈值的工具都作为候选，不在本规则决策阶段作为判断依据
        """
        if (not knowledge_resources_emb_recalled) or (
            len(knowledge_resources_lowly_relevant) == len(knowledge_resources_emb_recalled)
        ):
            # 如果没有绑定知识库 or 所有文档都是超低分，则直接进行无私域知识、无工具的通用回答
            return Decision.GENERAL_QA
        elif len(knowledge_resources_highly_relevant) > 0:
            # 如果存在超高分文档，则直接使用超高分文档进行回答
            return Decision.PRIVATE_QA
        else:
            # 其他情况：如果存在一些可能是 query【意图不明确】或【描述不清】导致的中间分相关文档，根据中分相关文档进行 query 重写
            return Decision.QUERY_CLARIFICATION

    def query_cls_pipeline(
        self, chat_history, query, llm, merge_query_cls_with_resp_or_rewrite, with_query_cls, **kwargs
    ):
        if merge_query_cls_with_resp_or_rewrite:
            if chat_history:
                result = self.query_cls_with_resp_or_rewrite(chat_history, query, llm, **kwargs)
                query_cls = result["query_cls"]
            else:
                # 如无history，目前处理成相当于开始一个新的话题
                query_cls = "new"

            if query_cls == "finish":
                # 如果是结束话题，则直接返回生成的回复
                return {
                    "status": IntentStatus.AGENT_FINISH_WITH_RESPONSE,
                    "response": result["response"],
                }
            elif query_cls == "continue":
                # 如果是继续话题，则其实也未必能直接复用上一轮会话的意图识别结果（除了）
                # 因为例如用户是在RAG意图下，但是改了原本输错的参数，则其实也是需要重新检索资源进行新的具体资源意图判断的
                # 因此，意图识别结果的复用其实最多到知识类、资源类这种级别的复用
                independent_query = result["rewritten_query"]
            elif query_cls == "new":
                # 如果是新的话题，则独立query即为输入的query
                independent_query = query
        else:
            if chat_history:
                if with_query_cls:
                    query_cls = self.latest_query_classification(chat_history, query, llm, **kwargs)
                else:
                    query_cls = "continue"
            else:
                query_cls = "new"

            if query_cls == "finish":
                return {
                    "status": IntentStatus.DIRECTLY_RESPOND_BY_AGENT,
                }
            elif query_cls == "continue":
                independent_query = self.query_rewrite_for_independence(chat_history, query, llm, **kwargs)
            elif query_cls == "new":
                independent_query = query

        return independent_query

    @retry(max_retries=5, max_seconds=3600)
    def sum_chat_history_for_query(self, chat_history, query, llm, **kwargs):
        if not chat_history:
            return None
        sys_prompt = self.__class__.intent_recognition_prompt_templates.get(
            "sum_chat_history_for_query_sys_prompt_template"
        )
        usr_prompt = self.__class__.intent_recognition_prompt_templates.get(
            "sum_chat_history_for_query_usr_prompt_template"
        ).render(chat_history=chat_history, query=query)
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=usr_prompt),
        ]
        conditional_dispatch_custom_event("custom_event", {"front_end_display": False}, **kwargs)
        invoke_func = invoke_decorator(llm.invoke, llm)
        resp = invoke_func(messages)
        conditional_dispatch_custom_event("custom_event", {"front_end_display": True}, **kwargs)
        resp_content = resp.content
        if resp_content.strip() == "None":
            return None
        else:
            return resp_content

    def independent_query_pipeline(
        self,
        independent_query,
        llm,
        tools,
        with_structured_data,
        with_index_specific_search,
        with_index_specific_search_init,
        with_index_specific_search_translation,
        with_index_specific_search_keywords,
        with_es_search_query,
        with_es_search_keywords,
        with_rrf,
        knowledge_items: list[dict],
        knowledge_bases: list[dict],
        knowledge_resource_rough_recall_topk,
        knowledge_resource_reject_threshold,
        knowledge_resource_fine_grained_score_type,
        self_query_threshold_top_n,
        do_tool_resource_retrieve,
        tool_resource_base_ids,
        tool_resource_rough_recall_topk,
        tool_resource_reject_threshold,
        tool_resource_fine_grained_score_type,
        gen_pseudo_tool_resource_desc,
        **kwargs,
    ):
        # ====================================================================================================
        # 资源召回和解析
        # ====================================================================================================
        # 知识类资源
        if knowledge_items or knowledge_bases:
            (
                knowledge_resources_emb_recalled,
                knowledge_resources_lowly_relevant,
                knowledge_resources_moderately_relevant,
                knowledge_resources_highly_relevant,
            ) = self.retrieve_and_parse_knowledge_resource(
                query=independent_query,
                llm=llm,
                with_structured_data=with_structured_data,
                with_index_specific_search=with_index_specific_search,
                with_index_specific_search_init=with_index_specific_search_init,
                with_index_specific_search_translation=with_index_specific_search_translation,
                with_index_specific_search_keywords=with_index_specific_search_keywords,
                with_es_search_query=with_es_search_query,
                with_es_search_keywords=with_es_search_keywords,
                with_rrf=with_rrf,
                knowledge_items=knowledge_items,
                knowledge_bases=knowledge_bases,
                knowledge_resource_rough_recall_topk=knowledge_resource_rough_recall_topk,
                knowledge_resource_reject_threshold=knowledge_resource_reject_threshold,
                knowledge_resource_fine_grained_score_type=knowledge_resource_fine_grained_score_type,
                self_query_threshold_top_n=self_query_threshold_top_n,
                **kwargs,
            )
        else:
            knowledge_resources_emb_recalled = None
            knowledge_resources_lowly_relevant = None
            knowledge_resources_moderately_relevant = None
            knowledge_resources_highly_relevant = None

        # 工具类资源
        if do_tool_resource_retrieve:
            (
                tool_resources_emb_recalled,
                tool_resources_lowly_relevant,
                tool_resources_moderately_relevant,
                tool_resources_highly_relevant,
            ) = self.retrieve_and_parse_tool_resource(
                query=independent_query,
                llm=llm,
                tool_resource_item_ids=[],
                tool_resource_base_ids=tool_resource_base_ids,
                tool_resource_rough_recall_topk=tool_resource_rough_recall_topk,
                tool_resource_reject_threshold=tool_resource_reject_threshold,
                tool_resource_fine_grained_score_type=tool_resource_fine_grained_score_type,
                gen_pseudo_tool_resource_desc=gen_pseudo_tool_resource_desc,
                **kwargs,
            )
            raise NotImplementedError("目前开发侧还不支持为工具类、agent类等资源提供注册表管理机制，并自动与知识库关联")

        else:
            # 如果提供的工具类资源少于tool_count_threshold则无需进行工具类资源召回，直接全量使用即可
            tool_resources_emb_recalled = None
            tool_resources_lowly_relevant = None
            tool_resources_moderately_relevant = None
            tool_resources_highly_relevant = None
            candidate_tools = copy.deepcopy(tools)

        # ====================================================================================================
        # 进行决策
        # ====================================================================================================
        decision = self.make_decision(
            knowledge_resources_emb_recalled,
            knowledge_resources_lowly_relevant,
            knowledge_resources_moderately_relevant,
            knowledge_resources_highly_relevant,
            tool_resources_emb_recalled,
            tool_resources_lowly_relevant,
            tool_resources_moderately_relevant,
            tool_resources_highly_relevant,
        )

        return {
            "status": IntentStatus.PROCESS_BY_AGENT,
            "decision": decision,
            "independent_query": independent_query,
            "candidate_tools": candidate_tools,
            "knowledge_resources_emb_recalled": knowledge_resources_emb_recalled,
            "knowledge_resources_highly_relevant": knowledge_resources_highly_relevant,
            "knowledge_resources_moderately_relevant": knowledge_resources_moderately_relevant,
        }

    @timeit(message="意图识别总流程")
    def exec_intent_recognition(
        self,
        query: str,
        llm: BaseChatModel,
        tools: List[BaseTool],
        callbacks: Callbacks = None,
        chat_history: List = None,
        intent_code: str = None,
        with_query_cls: bool = True,
        force_process_by_agent: bool = False,
        with_structured_data: bool = False,
        with_index_specific_search: bool = True,
        with_index_specific_search_init: bool = True,
        with_index_specific_search_translation: bool = False,
        with_index_specific_search_keywords: bool = False,
        with_es_search_query: bool = False,
        with_es_search_keywords: bool = False,
        with_rrf: bool = True,
        knowledge_items: List[dict] = [],
        knowledge_bases: List[dict] = [],
        knowledge_resource_rough_recall_topk: int = 10,
        knowledge_resource_reject_threshold: Tuple[float, float] = (0.0001, 0.1),
        knowledge_resource_fine_grained_score_type: FineGrainedScoreType = FineGrainedScoreType.LLM,
        self_query_threshold_top_n: int = 0,
        tool_resource_base_ids: List[int] = [],
        tool_resource_rough_recall_topk: int = 10,
        tool_resource_reject_threshold: Tuple[float, float] = (0.25, 0.75),
        tool_resource_fine_grained_score_type: FineGrainedScoreType = FineGrainedScoreType.EXCLUSIVE_SIMILARITY_MODEL,
        tool_count_threshold: int = 10000,
        gen_pseudo_tool_resource_desc: bool = True,
        retrieved_knowledge_resources: List = None,
        merge_query_cls_with_resp_or_rewrite: bool = False,
        independent_query_mode: IndependentQueryMode = IndependentQueryMode.SUM_AND_CONCATE,
        **kwargs,
    ):
        """
        基于多轮会话的意图识别主入口

        :param query: 用户最新输入的提问
        :param llm: 使用的 LLM
        :param tools: 非系统内置的候选工具列表
        :param callbacks: callback 列表
        :param chat_history: 多轮会话历史
        :param intent_code: 约定的意图识别 code，用于快速单跳
        :param with_query_cls: 是否进行意图切换检测。NOTE: 目前仅在非 merge_query_cls_with_resp_or_rewrite 的情况下才生效
        :param force_process_by_agent: 是否强制进入 IntentStatus.PROCESS_BY_AGENT 的 status。用于 AIDEV 产品页面召回测试
        :param with_structured_data: 用户勾选的知识中是否带结构化数据。NOTE: 目前该值为 True 时才会进行 nature 方式的知识召回
        :param with_index_specific_search: 是否使用基于 embedding 模型的 index specific 召回
        :param with_index_specific_search_init: 是否使用原始的 query，使用基于 embedding 模型的 index specific 召回
            注意仅在 query 确实被改写了才生效
        :param with_index_specific_search_translation: 是否在判断发现输入的 query 为英文的情况下，将其翻译成中文后再进行召回
        :param with_index_specific_search_keywords: 是否使用 query 提取的关键词，
            并使用基于 embedding 模型的 index specific 召回
        :param with_es_search_query: 是否使用原始 query 在 ES 上进行召回
        :param with_es_search_keywords: 是否使用 query 提取的关键词 在 ES 上进行召回
        :param with_rrf: 是否使用 weighted reciprocal rank fusion 对多路召回的结果进行融合。
            如果为 False，则直接合并后使用 __score__ 字段的值进行降序排序
        :param knowledgeitem_ids: 知识类资源 item ID 列表
        :param knowledgebase_ids: 知识类资源 base ID 列表
        :param knowledge_resource_rough_recall_topk: 知识类资源粗召 topk 值
        :param knowledge_resource_reject_threshold: 知识类资源拒答和直接回答的阈值
        :param knowledge_resource_fine_grained_score_type: 知识类资源进行细粒度的相似度计算的模型类型
        :param self_query_threshold_top_n: 在第 1 步意图识别判断用户 query 是否涉及结构化数据的时候，
            使用全量检索结果的 top_n 中是否包含结构化数据来判断。
            NOTE: 当前先将其值置为0，即不开启 self query 分支。
            TODO: 后续去除 nature 分支后，即可将其值置为 5 等，开启 self query 分支。
        :param tool_resource_base_ids: 工具类资源 base ID 列表。NOTE: 目前工具类资源统一放 base ID 中不放 item ID 中
        :param tool_resource_rough_recall_topk: 工具类资源粗召 topk 值
        :param tool_resource_reject_threshold: 工具类资源拒答和直接回答的阈值
        :param tool_resource_fine_grained_score_type: 工具类资源进行细粒度的相似度计算的模型类型
        :param tool_count_threshold: 当能够获取的工具的个数大于该阈值时，才开启工具类资源的粗召 + 精排，
            否则每次调用兜底LLM agent时都附上所有工具类资源
        :param gen_pseudo_tool_resource_desc: 在进行工具类资源的粗召时，是否进行伪工具类资源描述的生成
        :param retrieved_knowledge_resources: 用户自带的历史检索的上下文
        :param merge_query_cls_with_resp_or_rewrite: 是否将意图切换检测和 query 重写/直接答复合并在一次LLM调用中
        :param independent_query_mode: 独立 query 的构造逻辑
        """
        if callbacks:
            _set_config_context({"callbacks": callbacks})

        # ====================================================================================================
        # 如果是 force_process_by_agent 的情况下，则直接走知识类资源召回+精排的路，且获取 make decision 结果
        # ====================================================================================================
        if force_process_by_agent:
            return self.independent_query_pipeline(
                query,
                None,
                [],
                with_structured_data,
                with_index_specific_search,
                with_index_specific_search_init,
                with_index_specific_search_translation,
                with_index_specific_search_keywords,
                with_es_search_query,
                with_es_search_keywords,
                with_rrf,
                knowledge_items,
                knowledge_bases,
                knowledge_resource_rough_recall_topk,
                knowledge_resource_reject_threshold,
                knowledge_resource_fine_grained_score_type,
                self_query_threshold_top_n,
                False,
                None,
                None,
                None,
                None,
                None,
                **kwargs,
            )

        # ====================================================================================================
        # 如果带历史检索上下文，则说明用户的意图非常明确，需设置工具列表为空，且直接使用该上下文进行回答
        # ====================================================================================================
        if retrieved_knowledge_resources:
            return {
                "status": IntentStatus.QA_WITH_RETRIEVED_KNOWLEDGE_RESOURCES,
                "retrieved_knowledge_resources": retrieved_knowledge_resources,
            }

        # ====================================================================================================
        # 为【无需history的】且可通过3种特殊意图识别方式判断的query提供快速单跳通道
        # ====================================================================================================
        # NOTE:
        # 下述3种快速单跳通道，根据跳至的意图类别的不同，后续走的逻辑分支更不相同，需要每种都去特殊支持。例如：
        # 假设是跳至某个具体的工具，则直接调用LLM进行工具参数生成，然后调用工具即可
        # 假设是跳至某个智能工单查证场景这种特殊的通道，则调用对应的分类小模型，然后进行工单查证
        # 假设是跳至某个特殊的知识库索引，则可能是调用对应的索引进行召回，然后进行LLM问答
        if intent_code:
            intent = self.intent_recognition_by_code(intent_code)
            raise NotImplementedError("需要根据该具体的intent设定进行后续对应处理逻辑的实现")
        intent = self.intent_recognition_by_template(query)
        if intent:
            raise NotImplementedError("需要根据该具体的intent设定进行后续对应处理逻辑的实现")
        intent = self.intent_recognition_by_exclusive_model(query)
        if intent:
            raise NotImplementedError("需要根据该具体的intent设定进行后续对应处理逻辑的实现")

        # ====================================================================================================
        # 快速单跳通道的实际实现例子
        # ====================================================================================================
        intent = self.intent_recognition_by_template_one(query)
        if intent == "directly_respond":
            return {
                "status": IntentStatus.DIRECTLY_RESPOND_BY_AGENT,
            }

        # ====================================================================================================
        # 正常通道
        # ====================================================================================================
        # NOTE: 待开发侧支持为工具类、agent类等资源提供注册表管理机制，并自动与知识库关联后，
        # tool_resource_base_ids 与传入的 tools 会有关联，本判断条件需要更新
        # TODO: 待上述事项支持后，需要改成使用 exec_intent_recognition 传进来的参数
        do_tool_resource_retrieve = tool_resource_base_ids and len(tools) > tool_count_threshold
        # NOTE: 目前认为只有绑定了知识库，或者需要进行工具类资源召回的情况下，才可能需要进行 independent query 的改写
        # 此外，还加了 with_query_cls_and_rewrite 总开关
        res = query  # 默认初始化为 query
        if knowledge_items or knowledge_bases or do_tool_resource_retrieve:
            if independent_query_mode == IndependentQueryMode.REWRITE:
                res = self.query_cls_pipeline(
                    chat_history, query, llm, merge_query_cls_with_resp_or_rewrite, with_query_cls, **kwargs
                )
            elif independent_query_mode == IndependentQueryMode.SUM_AND_CONCATE:
                sum_res = self.sum_chat_history_for_query(chat_history, query, llm, **kwargs)
                if sum_res:
                    res = f"{sum_res}\n{query}"

        if isinstance(res, dict) and "status" in res:
            return res
        elif isinstance(res, str):
            independent_query = res
        # TODO: 对于触发改写的情形，可以考虑原始 query 和 independent_query 分 2 路召回，最终对结果进行合并
        return self.independent_query_pipeline(
            independent_query,
            llm,
            tools,
            with_structured_data,
            with_index_specific_search,
            with_index_specific_search_init,
            with_index_specific_search_translation,
            with_index_specific_search_keywords,
            with_es_search_query,
            with_es_search_keywords,
            with_rrf,
            knowledge_items,
            knowledge_bases,
            knowledge_resource_rough_recall_topk,
            knowledge_resource_reject_threshold,
            knowledge_resource_fine_grained_score_type,
            self_query_threshold_top_n,
            do_tool_resource_retrieve,
            tool_resource_base_ids,
            tool_resource_rough_recall_topk,
            tool_resource_reject_threshold,
            tool_resource_fine_grained_score_type,
            gen_pseudo_tool_resource_desc,
            **kwargs,
        )
