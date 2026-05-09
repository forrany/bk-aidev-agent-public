# -*- coding: utf-8 -*-
"""``AgentConfigFetcher`` 请求级缓存契约：

- 同 ``(app_code, version, username)`` 在同请求内只触发一次底层调用；
- 不同 key 各自独立调用底层；
- 缓存命中返回值是深拷贝，调用方 mutation 不影响下次命中；
- ``clear_request_cache`` 后再次调用会重新触发底层。
"""

from unittest.mock import MagicMock

import pytest
from aidev_bkplugin.services.agent_config import AgentConfigFetcher


@pytest.fixture(autouse=True)
def _clear_cache():
    """每个用例前后清理请求级缓存，避免线程复用导致跨用例污染。"""
    AgentConfigFetcher.clear_request_cache()
    yield
    AgentConfigFetcher.clear_request_cache()


@pytest.fixture
def mock_rm(monkeypatch):
    """拦截 ``resource_manager().retrieve_agent_config``，返回可控 dict。"""
    rm = MagicMock()
    rm.retrieve_agent_config.return_value = {
        "agent_type": "common",
        "prompt_setting": {"content": []},
    }
    monkeypatch.setattr(
        "aidev_bkplugin.services.agent_config.resource_manager",
        lambda: rm,
    )
    return rm


def test_same_kwargs_calls_backend_once(mock_rm):
    AgentConfigFetcher.get_info(username="alice", version="v1")
    AgentConfigFetcher.get_info(username="alice", version="v1")
    assert mock_rm.retrieve_agent_config.call_count == 1


def test_different_kwargs_call_backend_separately(mock_rm):
    AgentConfigFetcher.get_info(username="alice", version="v1")
    AgentConfigFetcher.get_info(username="bob", version="v1")
    AgentConfigFetcher.get_info(username="alice", version="v2")
    assert mock_rm.retrieve_agent_config.call_count == 3


def test_caller_mutation_does_not_pollute_next_hit(mock_rm):
    """模拟 ``views/agent.py`` 的 ``agent_info.pop("otel_info", ...)`` 场景。"""
    first = AgentConfigFetcher.get_info(username="alice")
    first.pop("agent_type", None)
    first["prompt_setting"]["content"].append({"role": "user", "content": "x"})

    second = AgentConfigFetcher.get_info(username="alice")
    assert second["agent_type"] == "common"
    assert second["prompt_setting"]["content"] == []


def test_clear_request_cache_forces_refetch(mock_rm):
    AgentConfigFetcher.get_info(username="alice")
    AgentConfigFetcher.clear_request_cache()
    AgentConfigFetcher.get_info(username="alice")
    assert mock_rm.retrieve_agent_config.call_count == 2


def test_get_role_info_inherits_cache(mock_rm):
    """``get_role_info`` 内部走 ``get_info``，自然命中同一缓存。"""
    mock_rm.retrieve_agent_config.return_value = {
        "agent_type": "common",
        "prompt_setting": {"content": [{"role": "system", "content": "hi"}]},
    }
    AgentConfigFetcher.get_info(username="alice")
    AgentConfigFetcher.get_role_info(username="alice")
    assert mock_rm.retrieve_agent_config.call_count == 1
