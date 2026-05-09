"""通用智能体子包门面。

外部统一从 ``aidev_agent.services.common_agent`` 导入：
- ``CommonQAAgent``：默认通用 agent 实现。
- ``CommonAgentProtocol``：通用 agent 实例契约。
- ``common_agent_factory``：实例工厂；plugin 通过 ``replace_defaults(MyAgent())`` 替换默认。
"""

from .agent import CommonQAAgent
from .registry import CommonAgentProtocol, common_agent_factory

__all__ = [
    "CommonAgentProtocol",
    "CommonQAAgent",
    "common_agent_factory",
]
