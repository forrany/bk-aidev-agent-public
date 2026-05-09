import json

import pytest
from aidev_agent.config import settings
from aidev_agent.core.tools.knowledge import make_knowledge_retrieval_tool
from aidev_agent.packages.langchain_core.models import ChatModel
from aidev_agent.pydantic_models import AgentOptions, KnowledgebaseSettings


@pytest.mark.skipif(
    not all([settings.APP_CODE, settings.SECRET_KEY]),
    reason="没有配置足够的环境变量,跳过该测试",
)
@pytest.mark.slow
def test_knowledge_tool():
    llm = ChatModel.get_setup_instance(model="qwen3-235B")
    agent_options = AgentOptions()
    with open("tests/mock_data/knowledgebase.json") as fi:
        knowledgebase = json.load(fi)
    agent_options.knowledge_query_options = KnowledgebaseSettings(
        knowledge_bases=[knowledgebase], knowledge_resource_rough_recall_topk=2, raw=True
    )
    tool = make_knowledge_retrieval_tool(llm, agent_options)
    result = tool.invoke(input={"query": "私有化部署deepseek"})
    print(result)
