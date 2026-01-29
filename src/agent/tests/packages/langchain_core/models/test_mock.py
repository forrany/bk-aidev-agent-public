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

import pytest
from aidev_agent.packages.langchain_core.models.mock import MockChatModel, MockEmbeddings, MockResponse
from langchain_core.messages import HumanMessage


class TestMockChatModel:
    """测试MockChatModel"""

    def test_simple_invoke(self):
        """测试简单的同步调用"""
        model = MockChatModel(responses=["Hello, World!"])
        messages = [HumanMessage(content="Hi")]

        response = model.invoke(messages)

        assert response is not None
        assert response.content == "Hello, World!"

    def test_multiple_responses(self):
        """测试多个响应循环使用"""
        model = MockChatModel(responses=["Response 1", "Response 2", "Response 3"])
        messages = [HumanMessage(content="Test")]

        # 第一次调用
        response1 = model.invoke(messages)
        assert response1.content == "Response 1"

        # 第二次调用
        response2 = model.invoke(messages)
        assert response2.content == "Response 2"

        # 第三次调用
        response3 = model.invoke(messages)
        assert response3.content == "Response 3"

        # 第四次调用，应该循环回到第一个
        response4 = model.invoke(messages)
        assert response4.content == "Response 1"

    def test_streaming(self):
        """测试流式输出"""
        model = MockChatModel(responses=["Hello World"], stream_chunk_size=5)
        messages = [HumanMessage(content="Hi")]

        chunks = list(model.stream(messages))

        # 验证收到了多个chunk
        assert len(chunks) > 1

        # 拼接所有chunk的内容
        full_content = "".join([chunk.content for chunk in chunks if chunk.content])
        assert full_content == "Hello World"

    @pytest.mark.asyncio
    async def test_async_invoke(self):
        """测试异步调用"""
        model = MockChatModel(responses=["Async response"])
        messages = [HumanMessage(content="Test")]

        response = await model.ainvoke(messages)

        assert response is not None
        assert response.content == "Async response"

    @pytest.mark.asyncio
    async def test_async_streaming(self):
        """测试异步流式输出"""
        model = MockChatModel(responses=["Async streaming"], stream_chunk_size=5)
        messages = [HumanMessage(content="Test")]

        chunks = []
        async for chunk in model.astream(messages):
            chunks.append(chunk)

        # 验证收到了多个chunk
        assert len(chunks) > 1

        # 拼接所有chunk的内容
        full_content = "".join([chunk.content for chunk in chunks if chunk.content])
        assert full_content == "Async streaming"

    def test_tool_calls(self):
        """测试工具调用"""
        model = MockChatModel(
            responses=[""], tool_calls=[[{"name": "get_weather", "args": {"city": "Beijing"}, "id": "call_1"}]]
        )
        messages = [HumanMessage(content="What's the weather?")]

        response = model.invoke(messages)

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "get_weather"
        assert response.tool_calls[0]["args"]["city"] == "Beijing"

    def test_reasoning_content(self):
        """测试推理内容（模拟deepseek-r1）"""
        model = MockChatModel(
            responses=["Final answer"], reasoning_contents=["Let me think... This is the reasoning process."]
        )
        messages = [HumanMessage(content="Solve this problem")]

        response = model.invoke(messages)

        assert response.content == "Final answer"
        assert "reasoning_content" in response.additional_kwargs
        assert response.additional_kwargs["reasoning_content"] == "Let me think... This is the reasoning process."

    def test_reasoning_content_streaming(self):
        """测试推理内容的流式输出"""
        model = MockChatModel(responses=["Final answer"], reasoning_contents=["Reasoning process"], stream_chunk_size=5)
        messages = [HumanMessage(content="Test")]

        chunks = list(model.stream(messages))
        has_reasoning = False
        has_reasoning_time = False

        for chunk in chunks:
            # stream返回的是AIMessage对象，直接访问additional_kwargs
            if hasattr(chunk, "additional_kwargs") and "reasoning_content" in chunk.additional_kwargs:
                has_reasoning = True
            if hasattr(chunk, "additional_kwargs") and "reasoning_time" in chunk.additional_kwargs:
                has_reasoning_time = True

        # 验证包含推理内容和推理时间
        assert has_reasoning
        assert has_reasoning_time

    def test_token_counting(self):
        """测试token计数"""
        model = MockChatModel()

        # 测试文本token计数
        text = "This is a test message"
        token_count = model.get_num_tokens(text)
        assert token_count > 0

        # 测试消息token计数
        messages = [
            HumanMessage(content="Hello"),
            HumanMessage(content="World"),
        ]
        message_token_count = model.get_num_tokens_from_messages(messages)
        assert message_token_count > 0

    def test_llm_type(self):
        """测试LLM类型"""
        model = MockChatModel()
        assert model._llm_type == "mock-chat-model"

    def test_mock_responses_with_different_types(self):
        """测试使用MockResponse按顺序返回不同类型的响应"""
        model = MockChatModel(
            mock_responses=[
                # 第一个响应：工具调用
                MockResponse(
                    content="",
                    tool_calls=[{"name": "get_weather", "args": {"city": "Beijing"}, "id": "call_1"}],
                ),
                # 第二个响应：普通文本（模拟工具结果的总结）
                MockResponse(content="The weather in Beijing is sunny, 25°C."),
                # 第三个响应：带推理内容
                MockResponse(
                    content="Based on the analysis, I recommend...",
                    reasoning_content="Let me think about this carefully...",
                ),
            ]
        )
        messages = [HumanMessage(content="What's the weather?")]

        # 第一次调用：应该返回工具调用
        response1 = model.invoke(messages)
        assert response1.content == ""
        assert response1.tool_calls is not None
        assert len(response1.tool_calls) == 1
        assert response1.tool_calls[0]["name"] == "get_weather"

        # 第二次调用：应该返回普通文本
        response2 = model.invoke(messages)
        assert response2.content == "The weather in Beijing is sunny, 25°C."
        assert response2.tool_calls is None or len(response2.tool_calls) == 0

        # 第三次调用：应该返回带推理内容的响应
        response3 = model.invoke(messages)
        assert response3.content == "Based on the analysis, I recommend..."
        assert "reasoning_content" in response3.additional_kwargs
        assert response3.additional_kwargs["reasoning_content"] == "Let me think about this carefully..."

        # 第四次调用：应该循环回到第一个
        response4 = model.invoke(messages)
        assert response4.tool_calls is not None
        assert response4.tool_calls[0]["name"] == "get_weather"

    def test_mock_responses_streaming(self):
        """测试MockResponse在流式输出中的表现"""
        model = MockChatModel(
            mock_responses=[
                MockResponse(
                    content="",
                    tool_calls=[{"name": "calculator", "args": {"expr": "2+2"}, "id": "call_1"}],
                ),
                MockResponse(content="The result is 4"),
            ],
            stream_chunk_size=5,
        )
        messages = [HumanMessage(content="Calculate 2+2")]

        # 第一次流式调用：应该返回工具调用
        chunks1 = list(model.stream(messages))

        # stream()返回的是AIMessageChunk对象，直接检查tool_call_chunks属性
        has_tool_call = any(hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks for chunk in chunks1)
        assert has_tool_call

        # 第二次流式调用：应该返回文本
        chunks2 = list(model.stream(messages))
        full_content = "".join([chunk.content for chunk in chunks2 if chunk.content])
        assert full_content == "The result is 4"

    @pytest.mark.asyncio
    async def test_mock_responses_async(self):
        """测试MockResponse的异步调用"""
        model = MockChatModel(
            mock_responses=[
                MockResponse(content="First response"),
                MockResponse(
                    content="",
                    tool_calls=[{"name": "search", "args": {"query": "test"}, "id": "call_1"}],
                ),
            ]
        )
        messages = [HumanMessage(content="Test")]

        # 第一次异步调用
        response1 = await model.ainvoke(messages)
        assert response1.content == "First response"

        # 第二次异步调用
        response2 = await model.ainvoke(messages)
        assert response2.tool_calls is not None
        assert response2.tool_calls[0]["name"] == "search"

    def test_backward_compatibility(self):
        """测试向后兼容性：旧的responses参数仍然可用"""
        model = MockChatModel(
            responses=["Response 1", "Response 2"],
            tool_calls=[[{"name": "tool1", "args": {}, "id": "call_1"}]],
        )
        messages = [HumanMessage(content="Test")]

        # 第一次调用
        response1 = model.invoke(messages)
        assert response1.content == "Response 1"
        assert response1.tool_calls is not None

        # 第二次调用（没有tool_calls，因为只配置了一个）
        response2 = model.invoke(messages)
        assert response2.content == "Response 2"


class TestMockEmbeddings:
    """测试MockEmbeddings"""

    def test_embed_query(self):
        """测试查询文本嵌入"""
        embeddings = MockEmbeddings(dimension=768)

        vector = embeddings.embed_query("test query")

        assert len(vector) == 768
        assert all(isinstance(v, float) for v in vector)

    def test_embed_documents(self):
        """测试文档列表嵌入"""
        embeddings = MockEmbeddings(dimension=512)

        texts = ["document 1", "document 2", "document 3"]
        vectors = embeddings.embed_documents(texts)

        assert len(vectors) == 3
        assert all(len(v) == 512 for v in vectors)

    def test_deterministic_embeddings(self):
        """测试相同文本生成相同的向量"""
        embeddings = MockEmbeddings(dimension=256)

        text = "test text"
        vector1 = embeddings.embed_query(text)
        vector2 = embeddings.embed_query(text)

        # 相同文本应该生成相同的向量
        assert vector1 == vector2

    @pytest.mark.asyncio
    async def test_async_embed_query(self):
        """测试异步查询嵌入"""
        embeddings = MockEmbeddings(dimension=384)

        vector = await embeddings.aembed_query("async test")

        assert len(vector) == 384

    @pytest.mark.asyncio
    async def test_async_embed_documents(self):
        """测试异步文档嵌入"""
        embeddings = MockEmbeddings(dimension=256)

        texts = ["doc 1", "doc 2"]
        vectors = await embeddings.aembed_documents(texts)

        assert len(vectors) == 2
        assert all(len(v) == 256 for v in vectors)


class TestMockIntegration:
    """测试Mock类与其他组件的集成"""

    def test_mock_chat_model_with_bind_tools(self):
        """测试MockChatModel与bind_tools的集成"""
        model = MockChatModel(responses=["I'll use the tool"])

        # 定义工具
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Calculate math expression",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ]

        # 绑定工具
        model_with_tools = model.bind_tools(tools)

        # 调用
        messages = [HumanMessage(content="Calculate 2+2")]
        response = model_with_tools.invoke(messages)

        assert response is not None
        assert response.content == "I'll use the tool"

    def test_mock_embeddings_similarity(self):
        """测试使用MockEmbeddings计算相似度"""
        embeddings = MockEmbeddings(dimension=128)

        # 获取两个文本的向量
        vector1 = embeddings.embed_query("machine learning")
        vector3 = embeddings.embed_query("banana")

        # 计算余弦相似度
        def cosine_similarity(v1, v2):
            dot_product = sum(a * b for a, b in zip(v1, v2))
            magnitude1 = sum(x * x for x in v1) ** 0.5
            magnitude2 = sum(x * x for x in v2) ** 0.5
            return dot_product / (magnitude1 * magnitude2)

        # 相同文本的相似度应该是1
        same_vector = embeddings.embed_query("machine learning")
        similarity_same = cosine_similarity(vector1, same_vector)
        assert abs(similarity_same - 1.0) < 0.0001

        # 不同文本的相似度应该不是1
        similarity_diff = cosine_similarity(vector1, vector3)
        assert abs(similarity_diff - 1.0) > 0.01
