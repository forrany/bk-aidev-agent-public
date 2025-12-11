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
from aidev_agent.config import settings
from aidev_agent.packages.langchain_core.models.llm_gateway import ChatModel
from langchain_core.messages import HumanMessage


# 测试模型列表
TEST_MODELS = ["hunyuan-turbo", "qwen3", "deepseek-v3", "deepseek-r1"]


@pytest.mark.skipif(
    not all([settings.LLM_GW_ENDPOINT, settings.APP_CODE, settings.SECRET_KEY]),
    reason="没有配置足够的环境变量,跳过该测试",
)
@pytest.mark.slow
@pytest.mark.parametrize("model_name", TEST_MODELS)
def test_chat_model_simple_invoke(model_name):
    """测试纯模型调用 - 简单的同步调用"""
    # 创建模型实例
    chat_model = ChatModel.get_setup_instance(model=model_name)

    # 简单的消息
    messages = [HumanMessage(content="你好，请用一句话介绍你自己")]

    # 调用模型
    response = chat_model.invoke(messages)

    # 验证响应
    assert response is not None
    assert response.content
    assert len(response.content) > 0
    print(f"\n[{model_name}] 响应: {response.content}")


@pytest.mark.skipif(
    not all([settings.LLM_GW_ENDPOINT, settings.APP_CODE, settings.SECRET_KEY]),
    reason="没有配置足够的环境变量,跳过该测试",
)
@pytest.mark.slow
@pytest.mark.parametrize("model_name", TEST_MODELS)
def test_chat_model_streaming(model_name):
    """测试纯模型调用 - 流式响应"""
    # 创建模型实例
    chat_model = ChatModel.get_setup_instance(model=model_name)

    # 简单的消息
    messages = [HumanMessage(content="请简单说一下 Python 的特点")]

    # 流式调用
    chunks = []
    for chunk in chat_model.stream(messages):
        chunks.append(chunk)
        assert chunk is not None

    # 验证响应
    assert len(chunks) > 0
    full_content = "".join([chunk.content for chunk in chunks if chunk.content])
    assert len(full_content) > 0
    print(f"\n[{model_name}] 流式响应完整内容长度: {len(full_content)}")


@pytest.mark.skipif(
    not all([settings.LLM_GW_ENDPOINT, settings.APP_CODE, settings.SECRET_KEY]),
    reason="没有配置足够的环境变量,跳过该测试",
)
@pytest.mark.slow
@pytest.mark.parametrize("model_name", TEST_MODELS)
def test_chat_model_with_tool_call(model_name):
    """测试工具调用 - bind_tools 和 tool_choice"""
    # 定义一个简单的工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，例如：北京、上海",
                        },
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    # 创建模型实例并绑定工具
    chat_model = ChatModel.get_setup_instance(model=model_name)
    chat_model_with_tools = chat_model.bind_tools(tools)

    # 构造会触发工具调用的消息
    messages = [HumanMessage(content="北京今天的天气怎么样？")]

    # 调用模型
    response = chat_model_with_tools.invoke(messages)

    # 验证响应
    assert response is not None

    # 检查是否有工具调用
    # 注意：不是所有模型在所有情况下都会调用工具，这取决于模型的行为
    # 这里我们只验证响应的基本结构
    print(f"\n[{model_name}] 工具调用测试响应类型: {type(response)}")
    print(f"[{model_name}] 响应内容: {response.content if response.content else '(空)'}")

    # 如果有 tool_calls，打印出来
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"[{model_name}] 工具调用: {response.tool_calls}")
        assert len(response.tool_calls) > 0
    elif hasattr(response, "additional_kwargs") and response.additional_kwargs.get("tool_calls"):
        print(f"[{model_name}] 工具调用 (additional_kwargs): {response.additional_kwargs['tool_calls']}")


@pytest.mark.skipif(
    not all([settings.LLM_GW_ENDPOINT, settings.APP_CODE, settings.SECRET_KEY]),
    reason="没有配置足够的环境变量,跳过该测试",
)
@pytest.mark.slow
@pytest.mark.parametrize("model_name", TEST_MODELS)
def test_chat_model_with_tool_streaming(model_name):
    """测试工具调用 - 流式响应"""
    # 定义工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "执行数学计算",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "数学表达式，例如：2+2",
                        },
                    },
                    "required": ["expression"],
                },
            },
        }
    ]

    # 创建模型实例并绑定工具
    chat_model = ChatModel.get_setup_instance(model=model_name)
    chat_model_with_tools = chat_model.bind_tools(tools)

    # 构造消息
    messages = [HumanMessage(content="计算 15 + 27 等于多少")]

    # 流式调用
    chunks = []
    for chunk in chat_model_with_tools.stream(messages):
        chunks.append(chunk)
        assert chunk is not None

    # 验证响应
    assert len(chunks) > 0
    print(f"\n[{model_name}] 工具流式调用收到 {len(chunks)} 个 chunks")

    # 检查是否有工具调用信息
    for i, chunk in enumerate(chunks):
        if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
            print(f"[{model_name}] Chunk {i} 包含工具调用: {chunk.tool_call_chunks}")


@pytest.mark.skipif(
    not all([settings.LLM_GW_ENDPOINT, settings.APP_CODE, settings.SECRET_KEY]),
    reason="没有配置足够的环境变量,跳过该测试",
)
@pytest.mark.slow
def test_chat_model_reasoning_content():
    """测试 deepseek-r1 的推理内容 (reasoning_content) 功能"""
    # 只测试 deepseek-r1，因为这是支持 reasoning_content 的模型
    model_name = "deepseek-r1"

    # 创建模型实例
    chat_model = ChatModel.get_setup_instance(model=model_name)

    # 构造需要推理的问题
    messages = [HumanMessage(content="请解释为什么 1+1=2")]

    # 调用模型
    response = chat_model.invoke(messages)

    # 验证响应
    assert response is not None
    print(f"\n[{model_name}] 响应内容: {response.content[:100] if response.content else '(空)'}...")

    # 检查是否有 reasoning_content
    if hasattr(response, "additional_kwargs") and "reasoning_content" in response.additional_kwargs:
        reasoning = response.additional_kwargs["reasoning_content"]
        print(f"[{model_name}] 推理内容: {reasoning[:100] if reasoning else '(空)'}...")
        assert reasoning is not None


@pytest.mark.skipif(
    not all([settings.LLM_GW_ENDPOINT, settings.APP_CODE, settings.SECRET_KEY]),
    reason="没有配置足够的环境变量,跳过该测试",
)
@pytest.mark.slow
def test_chat_model_token_counting():
    """测试 token 计数功能"""
    model_name = "hunyuan-turbo"

    # 创建模型实例
    chat_model = ChatModel.get_setup_instance(model=model_name, remote_tokenizer=True)

    # 测试文本 token 计数
    text = "这是一段测试文本，用于验证 token 计数功能是否正常工作"
    token_count = chat_model.get_num_tokens(text)

    assert token_count > 0
    print(f"\n[{model_name}] 文本 token 数量: {token_count}")

    # 测试消息 token 计数
    messages = [
        HumanMessage(content="你好"),
        HumanMessage(content="这是第二条消息"),
    ]
    message_token_count = chat_model.get_num_tokens_from_messages(messages)

    assert message_token_count > 0
    print(f"[{model_name}] 消息列表 token 数量: {message_token_count}")
