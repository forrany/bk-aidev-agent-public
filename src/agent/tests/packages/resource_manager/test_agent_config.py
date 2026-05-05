"""``BaseResourceManager.get_agent_config`` 契约测试。

覆盖：
- 每次调用都打到 ``retrieve_agent_config``（SDK 不做进程内缓存）；
- ``version`` 透传到 ``retrieve_agent_config``；
- 装配规则：``KnowledgebaseSettings`` 默认拒答文案、``prompt_setting`` 中
  ``llm_token_limit`` / ``tool_output_compress_thrd`` 合并到对应配置；
- ``retrieve_agent_config`` 抛错时统一包裹为 ``ValueError``。
"""

from __future__ import annotations

from typing import Optional

import pytest
from aidev_agent.packages.resource_manager.base import BaseResourceManager


def _build_raw(**overrides) -> dict:
    raw = {
        "agent_name": "Test Agent",
        "prompt_setting": {
            "llm_code": "test-model",
            "non_thinking_llm": "test-non-thinking-model",
            "content": [{"role": "system", "content": "Test role prompt"}],
        },
        "related_tools": ["tool1", "tool2"],
        "conversation_settings": {"opening_remark": "Hello!", "commands": []},
        "intent_recognition": {},
        "knowledgebase_settings": {"knowledgebases": []},
    }
    raw.update(overrides)
    return raw


class _StubResourceManager(BaseResourceManager):
    """覆盖 ``retrieve_agent_config`` 以避开 HTTP；不实现 ``get_client``。"""

    def __init__(self, raw_factory=None, raise_exc: Exception | None = None):
        super().__init__(app_code="x", app_secret="y")
        self.raw_factory = raw_factory or (lambda agent_code, version: _build_raw())
        self.call_count = 0
        self.last_version: Optional[str] = None
        self._raise_exc = raise_exc

    def retrieve_agent_config(self, agent_code: str, version: Optional[str] = None, **kwargs) -> dict:
        self.call_count += 1
        self.last_version = version
        if self._raise_exc is not None:
            raise self._raise_exc
        return self.raw_factory(agent_code, version)


def test_get_agent_config_calls_retrieve_each_time():
    rm = _StubResourceManager()
    cfg1 = rm.get_agent_config("a1")
    cfg2 = rm.get_agent_config("a1")
    assert rm.call_count == 2
    assert cfg1 is not cfg2
    assert cfg1.agent_code == "a1"
    assert cfg2.agent_code == "a1"


@pytest.mark.parametrize("version", ["v2", None])
def test_version_passthrough(version):
    rm = _StubResourceManager()
    rm.get_agent_config("agent", version=version)
    assert rm.last_version == version


def test_default_rejection_message_when_kb_unmatched():
    """未设置 ``is_response_when_no_knowledgebase_match`` + ``rejection_message`` 时回填默认文案。"""
    rm = _StubResourceManager()
    cfg = rm.get_agent_config("a")
    assert cfg.agent_options.knowledge_query_options.rejection_message
    assert "无法" in cfg.agent_options.knowledge_query_options.rejection_message


def test_prompt_setting_hyperparams_merged():
    """``llm_token_limit`` → KB；``tool_output_compress_thrd`` → IR。"""
    raw = _build_raw()
    raw["prompt_setting"].update({"llm_token_limit": 12345, "tool_output_compress_thrd": 999})
    rm = _StubResourceManager(raw_factory=lambda *_: raw)
    cfg = rm.get_agent_config("a")
    assert cfg.agent_options.knowledge_query_options.llm_token_limit == 12345
    assert cfg.agent_options.intent_recognition_options.tool_output_compress_thrd == 999


def test_retrieve_failure_wrapped_as_value_error():
    rm = _StubResourceManager(raise_exc=RuntimeError("boom"))
    with pytest.raises(ValueError, match="Failed to retrieve agent config"):
        rm.get_agent_config("a")
