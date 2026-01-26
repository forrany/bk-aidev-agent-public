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

import asyncio
import time
from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import BaseModel, Field


class MockChatModel(BaseChatModel):
    """用于测试的Mock ChatModel

    可以配置固定的响应内容，支持同步/异步调用和流式输出。

    示例:
        >>> # 简单使用
        >>> model = MockChatModel(responses=["Hello!", "How can I help?"])
        >>> result = model.invoke([HumanMessage(content="Hi")])
        >>> print(result.content)  # "Hello!"

        >>> # 流式输出
        >>> model = MockChatModel(responses=["Hello World"], stream_chunk_size=5)
        >>> for chunk in model.stream([HumanMessage(content="Hi")]):
        ...     print(chunk.content, end="")  # "Hello" " Worl" "d"

        >>> # 模拟工具调用
        >>> model = MockChatModel(
        ...     responses=[""],
        ...     tool_calls=[[{"name": "get_weather", "args": {"city": "Beijing"}, "id": "call_1"}]]
        ... )
        >>> result = model.invoke([HumanMessage(content="What's the weather?")])
        >>> print(result.tool_calls)  # [{"name": "get_weather", ...}]
    """

    responses: List[str] = Field(
        default_factory=lambda: ["Mock response"],
        description="预设的响应列表，会循环使用",
    )

    tool_calls: Optional[List[List[dict]]] = Field(
        default=None,
        description="预设的工具调用列表，与responses对应",
    )

    reasoning_contents: Optional[List[str]] = Field(
        default=None,
        description="预设的推理内容列表（用于模拟deepseek-r1等模型），与responses对应",
    )

    stream_chunk_size: int = Field(
        default=10,
        description="流式输出时每个chunk的字符数",
    )

    sleep_time: float = Field(
        default=0.0,
        description="每次调用的延迟时间（秒），用于模拟网络延迟",
    )

    current_index: int = Field(
        default=0,
        description="当前使用的响应索引",
    )

    class Config:
        arbitrary_types_allowed = True

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成响应"""
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)

        # 获取当前响应
        response_text = self.responses[self.current_index % len(self.responses)]

        # 创建消息
        message = AIMessage(content=response_text)

        # 添加工具调用
        if self.tool_calls and self.current_index < len(self.tool_calls):
            message.tool_calls = self.tool_calls[self.current_index]

        # 添加推理内容
        if self.reasoning_contents and self.current_index < len(self.reasoning_contents):
            message.additional_kwargs["reasoning_content"] = self.reasoning_contents[self.current_index]

        # 更新索引
        self.current_index = (self.current_index + 1) % len(self.responses)

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步生成响应"""
        if self.sleep_time > 0:
            await asyncio.sleep(self.sleep_time)

        return self._generate(messages, stop, run_manager, **kwargs)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """流式生成响应"""
        # 获取当前响应
        response_text = self.responses[self.current_index % len(self.responses)]

        # 先发送推理内容（如果有）
        if self.reasoning_contents and self.current_index < len(self.reasoning_contents):
            reasoning_text = self.reasoning_contents[self.current_index]
            # 分块发送推理内容
            for i in range(0, len(reasoning_text), self.stream_chunk_size):
                chunk_text = reasoning_text[i : i + self.stream_chunk_size]
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content="", additional_kwargs={"reasoning_content": chunk_text})
                )
                if self.sleep_time > 0:
                    time.sleep(self.sleep_time / 10)
                yield chunk

            # 发送推理结束标记
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    additional_kwargs={"reasoning_time": 1000},  # Mock时间
                )
            )
            yield chunk

        # 分块发送响应内容
        for i in range(0, len(response_text), self.stream_chunk_size):
            chunk_text = response_text[i : i + self.stream_chunk_size]
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=chunk_text))
            if self.sleep_time > 0:
                time.sleep(self.sleep_time / 10)
            yield chunk

        # 发送工具调用（如果有）
        if self.tool_calls and self.current_index < len(self.tool_calls):
            for tool_call in self.tool_calls[self.current_index]:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[tool_call]))
                yield chunk

        # 更新索引
        self.current_index = (self.current_index + 1) % len(self.responses)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """异步流式生成响应"""
        # 获取当前响应
        response_text = self.responses[self.current_index % len(self.responses)]

        # 先发送推理内容（如果有）
        if self.reasoning_contents and self.current_index < len(self.reasoning_contents):
            reasoning_text = self.reasoning_contents[self.current_index]
            # 分块发送推理内容
            for i in range(0, len(reasoning_text), self.stream_chunk_size):
                chunk_text = reasoning_text[i : i + self.stream_chunk_size]
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content="", additional_kwargs={"reasoning_content": chunk_text})
                )
                if self.sleep_time > 0:
                    await asyncio.sleep(self.sleep_time / 10)
                yield chunk

            # 发送推理结束标记
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    additional_kwargs={"reasoning_time": 1000},  # Mock时间
                )
            )
            yield chunk

        # 分块发送响应内容
        for i in range(0, len(response_text), self.stream_chunk_size):
            chunk_text = response_text[i : i + self.stream_chunk_size]
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=chunk_text))
            if self.sleep_time > 0:
                await asyncio.sleep(self.sleep_time / 10)
            yield chunk

        # 发送工具调用（如果有）
        if self.tool_calls and self.current_index < len(self.tool_calls):
            for tool_call in self.tool_calls[self.current_index]:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[tool_call]))
                yield chunk

        # 更新索引
        self.current_index = (self.current_index + 1) % len(self.responses)

    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "mock-chat-model"

    def get_num_tokens(self, text: str) -> int:
        """简单的token计数（按字符数/4估算）"""
        return len(text) // 4

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """计算消息列表的token数"""
        total = 0
        for message in messages:
            if isinstance(message.content, str):
                total += self.get_num_tokens(message.content)
        return total

    def bind_tools(
        self,
        tools: Any,
        **kwargs: Any,
    ) -> "MockChatModel":
        """绑定工具到模型

        对于Mock模型，这个方法只是返回自身的一个副本，
        实际的工具调用行为由tool_calls参数控制。
        """
        # 创建一个新的实例，保持所有配置
        return self.model_copy()


class MockEmbeddings(BaseModel):
    """用于测试的Mock Embeddings

    可以配置固定的向量维度和值，支持同步/异步调用。

    示例:
        >>> # 简单使用
        >>> embeddings = MockEmbeddings(dimension=768)
        >>> vectors = embeddings.embed_documents(["Hello", "World"])
        >>> print(len(vectors))  # 2
        >>> print(len(vectors[0]))  # 768

        >>> # 自定义向量值
        >>> embeddings = MockEmbeddings(dimension=3, default_value=0.5)
        >>> vector = embeddings.embed_query("test")
        >>> print(vector)  # [0.5, 0.5, 0.5]
    """

    dimension: int = Field(
        default=768,
        description="向量维度",
    )

    default_value: float = Field(
        default=0.1,
        description="向量的默认值",
    )

    sleep_time: float = Field(
        default=0.0,
        description="每次调用的延迟时间（秒），用于模拟网络延迟",
    )

    normalize: bool = Field(
        default=False,
        description="是否归一化向量",
    )

    class Config:
        arbitrary_types_allowed = True

    def _generate_embedding(self, text: str) -> List[float]:
        """生成一个embedding向量

        使用文本的哈希值来生成确定性的向量，这样相同的文本总是得到相同的向量
        """
        # 使用文本哈希值作为种子，生成确定性的向量
        import hashlib

        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)

        # 生成向量
        vector = []
        for i in range(self.dimension):
            # 使用哈希值和索引生成伪随机数
            value = ((hash_value + i * 12345) % 10000) / 10000.0
            # 缩放到 [-1, 1] 范围
            value = (value - 0.5) * 2
            vector.append(value)

        # 归一化（如果需要）
        if self.normalize:
            magnitude = sum(x * x for x in vector) ** 0.5
            if magnitude > 0:
                vector = [x / magnitude for x in vector]

        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)

        return [self._generate_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)

        return self._generate_embedding(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步嵌入文档列表"""
        if self.sleep_time > 0:
            await asyncio.sleep(self.sleep_time)

        return [self._generate_embedding(text) for text in texts]

    async def aembed_query(self, text: str) -> List[float]:
        """异步嵌入查询文本"""
        if self.sleep_time > 0:
            await asyncio.sleep(self.sleep_time)

        return self._generate_embedding(text)
