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


class MockResponse(BaseModel):
    """单个Mock响应的配置

    可以精确控制每次调用返回的内容类型。

    示例:
        >>> # 返回普通文本
        >>> response1 = MockResponse(content="Hello!")

        >>> # 返回带工具调用的响应
        >>> response2 = MockResponse(
        ...     content="",
        ...     tool_calls=[{"name": "get_weather", "args": {"city": "Beijing"}, "id": "call_1"}]
        ... )

        >>> # 返回带推理内容的响应
        >>> response3 = MockResponse(
        ...     content="Final answer",
        ...     reasoning_content="Let me think..."
        ... )
    """

    content: str = Field(
        default="",
        description="响应的文本内容",
    )

    tool_calls: Optional[List[dict]] = Field(
        default=None,
        description="工具调用列表",
    )

    reasoning_content: Optional[str] = Field(
        default=None,
        description="推理内容（用于模拟deepseek-r1等模型）",
    )

    class Config:
        arbitrary_types_allowed = True


class MockChatModel(BaseChatModel):
    """用于测试的Mock ChatModel

    可以配置固定的响应内容，支持同步/异步调用和流式输出。

    示例:
        >>> # 简单使用（字符串列表）
        >>> model = MockChatModel(responses=["Hello!", "How can I help?"])
        >>> result = model.invoke([HumanMessage(content="Hi")])
        >>> print(result.content)  # "Hello!"

        >>> # 使用MockResponse对象列表（推荐）
        >>> model = MockChatModel(mock_responses=[
        ...     MockResponse(
        ...         content="",
        ...         tool_calls=[{"name": "get_weather", "args": {"city": "Beijing"}, "id": "call_1"}]
        ...     ),
        ...     MockResponse(content="The weather in Beijing is sunny, 25°C")
        ... ])
        >>> # 第一次调用返回工具调用
        >>> result1 = model.invoke([HumanMessage(content="What's the weather?")])
        >>> print(result1.tool_calls)  # [{"name": "get_weather", ...}]
        >>> # 第二次调用返回文本响应
        >>> result2 = model.invoke([HumanMessage(content="...")])
        >>> print(result2.content)  # "The weather in Beijing is sunny, 25°C"

        >>> # 流式输出
        >>> model = MockChatModel(responses=["Hello World"], stream_chunk_size=5)
        >>> for chunk in model.stream([HumanMessage(content="Hi")]):
        ...     print(chunk.content, end="")  # "Hello" " Worl" "d"
    """

    model_name: str = Field(default="mock", description="模型名称")

    mock_responses: Optional[List[MockResponse]] = Field(
        default=None,
        description="预设的MockResponse对象列表（推荐使用），会循环使用",
    )

    responses: Optional[List[str]] = Field(
        default=None,
        description="预设的响应列表（向后兼容），会循环使用",
    )

    tool_calls: Optional[List[List[dict]]] = Field(
        default=None,
        description="预设的工具调用列表，与responses对应（向后兼容）",
    )

    reasoning_contents: Optional[List[str]] = Field(
        default=None,
        description="预设的推理内容列表（用于模拟deepseek-r1等模型），与responses对应（向后兼容）",
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

    loop: bool = Field(
        default=True,
        description="是否循环使用响应列表。如果为False，用完所有响应后返回空响应",
    )

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        """初始化，确保至少有一个响应配置"""
        super().__init__(**data)

        # 如果没有配置任何响应，使用默认值
        if self.mock_responses is None and self.responses is None:
            self.responses = ["Mock response"]

    def _get_current_response(self) -> MockResponse:
        """获取当前的响应配置

        优先使用mock_responses，如果没有则从旧的配置构建MockResponse
        """
        if self.mock_responses:
            # 使用新的MockResponse配置
            if self.loop:
                index = self.current_index % len(self.mock_responses)
            else:
                # 不循环：如果索引超出范围，返回空响应
                if self.current_index >= len(self.mock_responses):
                    return MockResponse(content="")
                index = self.current_index
            return self.mock_responses[index]
        else:
            # 向后兼容：从旧的配置构建MockResponse
            if self.loop:
                index = self.current_index % len(self.responses)
            else:
                # 不循环：如果索引超出范围，返回空响应
                if self.current_index >= len(self.responses):
                    return MockResponse(content="")
                index = self.current_index
            response = MockResponse(content=self.responses[index])

            # 添加工具调用
            if self.tool_calls and index < len(self.tool_calls):
                response.tool_calls = self.tool_calls[index]

            # 添加推理内容
            if self.reasoning_contents and index < len(self.reasoning_contents):
                response.reasoning_content = self.reasoning_contents[index]

            return response

    def _get_response_count(self) -> int:
        """获取响应总数"""
        if self.mock_responses:
            return len(self.mock_responses)
        else:
            return len(self.responses)

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

        # 获取当前响应配置
        current_response = self._get_current_response()

        # 创建消息
        message = AIMessage(content=current_response.content)

        # 添加工具调用
        if current_response.tool_calls:
            message.tool_calls = current_response.tool_calls

        # 添加推理内容
        if current_response.reasoning_content:
            message.additional_kwargs["reasoning_content"] = current_response.reasoning_content

        # 更新索引
        if self.loop:
            self.current_index = (self.current_index + 1) % self._get_response_count()
        else:
            self.current_index += 1

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
        # 获取当前响应配置
        current_response = self._get_current_response()

        # 先发送推理内容（如果有）
        if current_response.reasoning_content:
            reasoning_text = current_response.reasoning_content
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
        response_text = current_response.content
        for i in range(0, len(response_text), self.stream_chunk_size):
            chunk_text = response_text[i : i + self.stream_chunk_size]
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=chunk_text))
            if self.sleep_time > 0:
                time.sleep(self.sleep_time / 10)
            yield chunk

        # 发送工具调用（如果有）
        if current_response.tool_calls:
            import json

            for tool_call in current_response.tool_calls:
                # 转换为tool_call_chunk格式（args需要是JSON字符串）
                tool_call_chunk = {
                    "name": tool_call.get("name"),
                    "args": json.dumps(tool_call.get("args", {})),
                    "id": tool_call.get("id"),
                    "index": tool_call.get("index"),
                }
                chunk = ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[tool_call_chunk]))
                yield chunk

        # 更新索引
        if self.loop:
            self.current_index = (self.current_index + 1) % self._get_response_count()
        else:
            self.current_index += 1

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """异步流式生成响应"""
        # 获取当前响应配置
        current_response = self._get_current_response()

        # 先发送推理内容（如果有）
        if current_response.reasoning_content:
            reasoning_text = current_response.reasoning_content
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
        response_text = current_response.content
        for i in range(0, len(response_text), self.stream_chunk_size):
            chunk_text = response_text[i : i + self.stream_chunk_size]
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=chunk_text))
            if self.sleep_time > 0:
                await asyncio.sleep(self.sleep_time / 10)
            yield chunk

        # 发送工具调用（如果有）
        if current_response.tool_calls:
            import json

            for tool_call in current_response.tool_calls:
                # 转换为tool_call_chunk格式（args需要是JSON字符串）
                tool_call_chunk = {
                    "name": tool_call.get("name"),
                    "args": json.dumps(tool_call.get("args", {})),
                    "id": tool_call.get("id"),
                    "index": tool_call.get("index"),
                }
                chunk = ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[tool_call_chunk]))
                yield chunk

        # 更新索引
        if self.loop:
            self.current_index = (self.current_index + 1) % self._get_response_count()
        else:
            self.current_index += 1

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

        对于Mock模型，这个方法直接返回自身，
        以保持current_index状态在多次调用之间共享。
        实际的工具调用行为由tool_calls参数控制。
        """
        # 直接返回自身，不创建副本，以保持状态
        return self


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
