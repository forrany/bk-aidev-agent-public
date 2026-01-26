import pytest
from aidev_agent.api.bk_aidev import BKAidevApi
from aidev_agent.config import settings
from aidev_agent.packages.langchain_core.models.llm_gateway import ChatModel
from aidev_agent.packages.langchain_core.tools.base import make_mcp_tools
from aidev_agent.services.chat import ChatCompletionAgent, ExecuteKwargs
from aidev_agent.services.pydantic_models import (
    ChatPrompt,
)
from bkapi_client_core.client import json
from langchain_core.tools import tool


@tool
def get_weather(location: str) -> str:
    """获取指定地点的天气预报"""
    return f"天气预报：{location}, 多云，25度，湿度60%。"


@tool
def get_weather_error(location: str) -> str:
    """获取指定地点的天气预报"""
    raise ValueError("天气预报获取失败")


@pytest.fixture
def add_session():
    client = BKAidevApi.get_client()
    session_code = "onlyfortest1"
    client.api.create_chat_session(json={"session_code": session_code, "session_name": "testonly"})
    # 添加一些session content
    client.api.create_chat_session_content(
        json={
            "session_code": session_code,
            "role": "user",
            "content": "明天深圳天气怎么样?",
            "status": "success",
        }
    )
    yield session_code
    result = client.api.get_chat_session_contents(params={"session_code": session_code})
    for each in result.get("data", []):
        _id = each["id"]
        client.api.destroy_chat_session_content(path_params={"id": _id})
    client.api.destroy_chat_session(path_params={"session_code": session_code})


@pytest.mark.skipif(
    not all([settings.APP_CODE, settings.SECRET_KEY]),
    reason="没有配置足够的环境变量,跳过该测试",
)
@pytest.mark.slow
class TestCommonAgentChatStreaming:
    """测试聊天代理的流式响应功能"""

    def setup_method(self):
        """每个测试方法执行前的设置"""
        self.llm = ChatModel.get_setup_instance(model="qwen3-235B")

    def test_basic_chat(self):
        """case 1: 基础聊天测试"""
        agent = ChatCompletionAgent(
            chat_model=self.llm,
            chat_history=[
                ChatPrompt(
                    id="1",
                    role="system",
                    content="You are a professional translator, please help translate the user input to English.",
                ),
                ChatPrompt(id="2", role="user", content="안녕하세요"),
                ChatPrompt(id="3", role="assistant", content="Hello, how can I help you?"),
                ChatPrompt(id="4", role="user", content="复述一下上下文的内容"),
            ],
        )
        result = agent.execute(ExecuteKwargs(stream=True))
        with open("text.log", "w") as fo:
            for each in result:
                fo.write(each)

    def test_tool_calling(self):
        """case 2: 工具调用"""
        llm = ChatModel.get_setup_instance(model="deepseek-r1")
        agent = ChatCompletionAgent(
            chat_model=llm,
            chat_history=[
                ChatPrompt(id="1", role="user", content="今天广州天气怎么样？"),
            ],
            tools=[get_weather],
        )
        result = agent.execute(ExecuteKwargs(stream=True))
        with open("text.log", "w") as fo:
            for each in result:
                fo.write(each)

    def test_resume_from_checkpoint(self):
        """case 3: 断点续传"""
        agent = ChatCompletionAgent(
            thread_id="onlyfortest",
            chat_model=self.llm,
            chat_history=[
                ChatPrompt(role="user", content="今天广州天气怎么样？"),
            ],
            tools=[get_weather],
        )
        with open("text.log", "w") as fo:
            result = agent.execute(ExecuteKwargs(stream=True))
            for idx, each in enumerate(result):
                if idx == 10:
                    break
                fo.write(each)

        agent2 = ChatCompletionAgent(
            thread_id="onlyfortest",
            chat_model=self.llm,
            chat_history=[
                ChatPrompt(role="user", content="今天广州天气怎么样？"),
            ],
            tools=[get_weather],
        )
        with open("text.log", "a") as fo:
            for each in agent2.execute(ExecuteKwargs(stream=True)):
                fo.write(each)

    def test_knowledge_base(self):
        """case 4: 知识库"""
        with open("tests/mock_data/knowledgebase.json") as fi:
            knowledgebase = json.load(fi)
        agent = ChatCompletionAgent(
            chat_model=self.llm,
            chat_history=[
                ChatPrompt(role="user", content="如何私有化部署deepseek"),
                # ChatPrompt(role="user", content="云桌面黑屏如何处理?"),
            ],
            knowledge_bases=[knowledgebase],
        )
        with open("text.log", "w") as fo:
            result = agent.execute(ExecuteKwargs(stream=True))
            for each in result:
                fo.write(each)

    def test_model_error_case(self):
        """case 5: 模型错误处理

        测试当使用无权限的模型时，错误信息能够被正确捕获并返回给消费者。
        错误响应格式: data: {"event": "error", "code": "UNKNOWN", "message": "模型调用异常: ..."}
        """
        import json

        # 设置一个不存在的模型
        self.llm = ChatModel.get_setup_instance(model="gptoss-999b")
        agent = ChatCompletionAgent(
            chat_model=self.llm,
            chat_history=[
                ChatPrompt(role="user", content="nonono"),
            ],
        )
        result_content = []
        result = agent.execute(ExecuteKwargs(stream=True))
        result_content = list(result)

        # 验证错误消息被正确捕获
        # 响应格式: data: {"event": "error", "code": "...", "message": "..."}
        last_content = result_content[-1]
        assert last_content.startswith("data: ")
        assert json.loads(last_content[5:])["type"] == "RUN_ERROR"
        assert json.loads(last_content[5:])["message"].startswith("模型调用异常:")

    def test_tool_call_error_case(self):
        """case 6: 工具调用错误处理"""

        # 设置一个不存在的模型
        self.llm = ChatModel.get_setup_instance(model="deepseek-r1")
        agent = ChatCompletionAgent(
            chat_model=self.llm,
            chat_history=[
                ChatPrompt(role="user", content="今天深圳天气"),
            ],
            tools=[get_weather_error],
        )
        result = agent.execute(ExecuteKwargs(stream=True))
        with open("text.log", "w") as fo:
            for each in result:
                fo.write(each)

    def test_mcp_calling(self):
        """case 7: MCP 调用"""
        tools = make_mcp_tools(
            {
                "tool-mcp-time": {
                    "url": "http://127.0.0.1:18888/mcp",
                    "transport": "streamable_http",
                }
            }
        )
        agent = ChatCompletionAgent(
            chat_model=self.llm,
            chat_history=[ChatPrompt(id="1", role="user", content="今天深圳天气怎么样?")],
            tools=tools,
        )
        result = agent.execute(ExecuteKwargs(stream=True))
        with open("text.log", "w") as fo:
            for each in result:
                fo.write(each)
