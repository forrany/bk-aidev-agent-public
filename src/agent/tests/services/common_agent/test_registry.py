# -*- coding: utf-8 -*-
"""``common_agent_factory`` 与 ``CommonAgentProtocol`` 单元测试。

覆盖：
- 默认 ``common_agent_factory.get()`` 返回 ``CommonQAAgent`` 实例（取消 Teapot 兜底）；
- ``replace_defaults(SubInstance)`` 后 ``get()`` 返回该实例，并返回旧默认；
- 仅实现 ``get_agent_executor`` 的 plugin 风格实例与 ``CommonAgentProtocol`` 结构兼容。
"""

import pytest
from aidev_agent.services.common_agent import (
    CommonAgentProtocol,
    CommonQAAgent,
    common_agent_factory,
)


@pytest.fixture
def restore_default():
    """每个用例运行前后保留 / 还原 factory 默认实例，避免相互污染。"""
    original = common_agent_factory.defaults
    yield
    common_agent_factory.replace_defaults(original)


class _PluginStyleAgent:
    """仅实现 ``get_agent_executor`` 的 plugin 风格实例，验证 Protocol 结构兼容。"""

    def __init__(self, marker: str = "plugin"):
        self.marker = marker

    def get_agent_executor(self, **kwargs):
        return ("graph", {"marker": self.marker, "kwargs": kwargs})


class TestCommonAgentFactory:
    def test_default_returns_common_qa_agent_instance(self):
        instance = common_agent_factory.get()
        assert isinstance(instance, CommonQAAgent)

    def test_call_without_key_equals_get(self):
        assert common_agent_factory() is common_agent_factory.get()

    def test_replace_defaults_returns_legacy_and_overrides_get(self, restore_default):
        legacy = common_agent_factory.defaults
        new_instance = _PluginStyleAgent(marker="replaced")

        returned_legacy = common_agent_factory.replace_defaults(new_instance)

        assert returned_legacy is legacy
        assert common_agent_factory.get() is new_instance


class TestCommonAgentProtocolCompat:
    @pytest.mark.parametrize(
        "instance",
        [CommonQAAgent(), _PluginStyleAgent()],
    )
    def test_instances_are_structurally_compatible(self, instance):
        agent: CommonAgentProtocol = instance
        assert callable(agent.get_agent_executor)
