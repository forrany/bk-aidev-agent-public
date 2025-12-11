from typing import ClassVar, Dict, TypedDict

from aidev_agent.core.graphs.react.graph import ReActAgent

class CommonQAAgent:
    """
    普通用户直接使用 CommonQAAgent 即可，会进行 agent 自适应路由
    高级用户需根据使用情况继承不同的 agent，并在 CommonQAAgent 中注册使用
    NOTE: 这里先继承自 ToolCallingCommonQAAgent，因为 aidev.resource.chat_completion.logic.ChatCompletionApp.get_window
    中需要使用到 ensure_memory_window。待开发侧确认各类需要使用 CommonQAAgent 成员函数/属性的场景。
    """

    agent_classes: ClassVar[Dict] = { }

    @classmethod
    def register_agent_class(cls, key, agent_class):
        cls.agent_classes[key] = agent_class

    @classmethod
    def get_agent_executor(cls, *args, **kwargs):
        agent, cfg = ReActAgent.get_agent_executor(*args, **kwargs)
        # agent_class = LangGraphV1QAAgent
        # class MyAgentState(TypedDict):
        #     input: str
        #
        # builder = AgentGraphBuilder()
        # builder = builder.with_model(
        #     ChatModel.get_setup_instance(model="qwen3"),
        # ).with_state_schema(MyAgentState)
        # agent, cfg = builder.build()
        # agent.agent = AgentStreamAdapter()
        # print("CommonQAAgent", agent)
        return agent, cfg
