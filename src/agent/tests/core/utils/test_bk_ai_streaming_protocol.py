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

from aidev_agent.enums import StreamEventType as EventType
from aidev_agent.packages.langgraph.streaming.streaming_protocol import BKAiStreamingAgentType, BkAiStreamingProtocol
from langchain_core.messages import AIMessageChunk


def test_process_event_on_chat_model_stream_with_tool_calls():
    """测试处理 on_chat_model_stream 事件，包含工具调用"""
    protocol = BkAiStreamingProtocol(timeout=30, skip_thought=False)
    protocol.agent_type = BKAiStreamingAgentType.ToolCallingCommonQAAgent

    chunk = AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": "chatcmpl-tool-fbfec876ea8e4584be5a88fb7d98c78e",
                    "function": {
                        "arguments": '{"name": "list_pods"}',
                        "name": "list_pods",
                    },
                    "type": "function",
                }
            ]
        },
        response_metadata={},
        id="run--047605e5-2f28-4d50-b992-ae3173da6eb1",
        tool_calls=[
            {
                "name": "list_pods",
                "args": {},
                "id": "chatcmpl-tool-fbfec876ea8e4584be5a88fb7d98c78e",
                "type": "tool_call",
            }
        ],
        tool_call_chunks=[
            {
                "name": "list_pods",
                "args": None,
                "id": "chatcmpl-tool-fbfec876ea8e4584be5a88fb7d98c78e",
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
    )

    test_event = {
        "event": "on_chat_model_stream",
        "data": {"chunk": chunk},
        "run_id": "047605e5-2f28-4d50-b992-ae3173da6eb1",
        "name": "ChatModel",
        "tags": ["seq:step:3"],
        "metadata": {
            "langgraph_step": 4,
            "langgraph_node": "analyze",
            "langgraph_triggers": ("branch:to:analyze",),
            "langgraph_path": ("__pregel_pull", "analyze"),
            "langgraph_checkpoint_ns": "analyze:558c5ac4-fdc7-8e54-c715-c09ec12b9ca6",
            "checkpoint_ns": "analyze:558c5ac4-fdc7-8e54-c715-c09ec12b9ca6",
            "ls_provider": "openai",
            "ls_model_name": "qwen3",
            "ls_model_type": "chat",
            "ls_temperature": 0.001,
        },
        "parent_ids": [
            "11963c69-2598-4ecf-b1e9-1fcebab7dc46",
            "9f20dbb8-e46a-4cbb-aadb-f51943b2caca",
            "702473fd-d504-4dfd-a81d-21c169467fc6",
        ],
    }

    events = list(protocol.process_event(test_event))
    print(events)

    assert len(events) > 0, "应该生成至少一个事件"
    first_event = events[0]
    assert first_event["event"] == EventType.THINK
    assert "list_pods" in first_event["content"]
    assert '"action": "list_pods"' in first_event["content"]

    run_id = test_event["run_id"]
    assert run_id in protocol.run_info
    assert protocol.run_info[run_id]["tool_call"] is True
    assert protocol.has_tool_call is True
    assert protocol.first_chunk is False


def test_process_event_empty_event():
    """测试处理空事件"""
    from aidev_agent.utils import Empty

    protocol = BkAiStreamingProtocol()

    events = list(protocol.process_event(Empty))
    assert len(events) > 0
    first_event = events[0]
    assert first_event["event"] == EventType.TEXT
    assert first_event["content"] == protocol.LOADING_AGENT_MESSAGE


def test_process_event_reasoning_content():
    """测试处理包含推理内容的事件"""
    protocol = BkAiStreamingProtocol()

    chunk = AIMessageChunk(
        content="",
        additional_kwargs={
            "reasoning_content": "正在思考如何回答这个问题...",
        },
        response_metadata={},
        id="run--test-id",
    )

    test_event = {
        "event": "on_chat_model_stream",
        "data": {"chunk": chunk},
        "run_id": "test-run-id",
    }

    events = list(protocol.process_event(test_event))
    assert len(events) > 0
    first_event = events[0]
    assert first_event["event"] == EventType.THINK
    assert first_event["content"] == "正在思考如何回答这个问题..."
    assert protocol.has_reasoning_content is True
