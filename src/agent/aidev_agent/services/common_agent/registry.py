"""通用智能体注册中心。

约束：
- ``CommonAgentProtocol`` 只约束「通用 agent 实例」对外暴露的 ``get_agent_executor`` 行为，
  使用 :class:`typing.Protocol`，结构化兼容（不做 isinstance 校验）。
- ``common_agent_factory`` 是 :class:`SingletonFactory` 实例，注册的是「实例」而非类，
  默认实例为 ``CommonQAAgent()``；plugin 通过 ``replace_defaults(MyAgent())`` 替换默认。
- 取用统一为 ``common_agent_factory.get()`` / ``common_agent_factory()``，无 key 调用即返回 default。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Tuple, runtime_checkable

from aidev_agent.utils.factory import SingletonFactory

from .agent import CommonQAAgent

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable, RunnableConfig


@runtime_checkable
class CommonAgentProtocol(Protocol):
    """通用智能体实例契约。

    实现者需暴露 ``get_agent_executor`` 行为（接收来自 ChatCompletionAgent 的标准化 kwargs，
    返回 ``(compiled_graph, runnable_config)``）。``CommonQAAgent.get_agent_executor`` 当前是
    ``@classmethod``，对实例调用同样有效，与本协议结构兼容。
    """

    def get_agent_executor(self, **kwargs) -> Tuple["Runnable", "RunnableConfig"]: ...


common_agent_factory: SingletonFactory[str, CommonAgentProtocol] = SingletonFactory(
    "common_agent",
    defaults=CommonQAAgent(),
)
