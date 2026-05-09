# -*- coding: utf-8 -*-
"""Agent 配置读取。

通过 ``resource_manager()``（``ResourceManagerProtocol`` 全局工厂）取回 agent 配置原始字典，
不直接依赖 ``aidev_agent.api.bk_aidev.BKAidevApi``。

请求级缓存：``get_info`` 在同一次 HTTP 请求内对相同 ``(app_code, version, username)`` 三元组只
触发一次远程调用；缓存挂在 ``aidev_agent.utils.local.request_local``（werkzeug ``Local``，
由 blueapps 中间件在请求结束时统一清理）。命中 / 写入均做 ``deepcopy``，确保调用方对返回
``dict`` 的 mutation（如 ``pop("otel_info", None)``）不影响后续命中结果。
"""

from __future__ import annotations

import base64
import copy
import json
from logging import getLogger
from typing import Optional

from aidev_agent.packages.resource_manager import resource_manager
from aidev_agent.pydantic_models import ChatPrompt
from aidev_agent.utils.local import request_local

logger = getLogger(__name__)

_REQUEST_CACHE_ATTR = "_agent_config_fetcher_cache"


class AgentConfigFetcher:
    """Plugin 层 agent 配置读取服务（无状态）。

    两个 ``@classmethod`` 调用入口：``get_info`` 取原始字典（``otel_info`` 自动 ``base64+json`` 解码），
    ``get_role_info`` 取装配后的 role prompts。所有过滤维度（``app_code`` / ``version`` / ``username``）
    通过 keyword-only 参数显式传入；``app_code`` 缺省回落到 ``settings.APP_CODE``。

    ``get_info`` 内置请求级缓存；``get_role_info`` 内部走 ``get_info``，自然继承缓存。
    """

    @classmethod
    def get_info(
        cls,
        *,
        app_code: Optional[str] = None,
        version: Optional[str] = None,
        username: Optional[str] = None,
    ) -> dict:
        """读取 agent 配置原始字典；``otel_info`` 自动解码。

        请求级缓存命中：直接返回缓存值的 ``deepcopy``。脱离请求上下文（如离线脚本）时降级为不缓存。
        """
        effective_app_code = app_code or resource_manager().get_agent_code()
        cache_key = (effective_app_code, version, username)
        cache = cls._get_request_cache()
        if cache is not None and cache_key in cache:
            return copy.deepcopy(cache[cache_key])

        agent_info = resource_manager().retrieve_agent_config(
            agent_code=effective_app_code,
            version=version,
            headers={"X-BKAIDEV-USER": username},
        )
        otel_env_info = agent_info.pop("otel_info", None)
        if otel_env_info:
            agent_info["otel_info"] = json.loads(base64.b64decode(otel_env_info).decode())

        if cache is not None:
            cache[cache_key] = copy.deepcopy(agent_info)
        return agent_info

    @classmethod
    def get_role_info(
        cls,
        *,
        app_code: Optional[str] = None,
        version: Optional[str] = None,
        username: Optional[str] = None,
    ) -> list[ChatPrompt]:
        """读取 ``prompt_setting.content``；``hidden-`` 前缀剥掉、``pause`` 映射为 ``assistant``。"""
        agent_config_info = cls.get_info(app_code=app_code, version=version, username=username)
        agent_role_content = agent_config_info["prompt_setting"].get("content", [])
        if not agent_role_content:
            return []

        for each in agent_role_content:
            each["role"] = each["role"].replace("hidden-", "")
            if each["role"] == "pause":
                each["role"] = "assistant"

        return [ChatPrompt(role=each["role"], content=each["content"]) for each in agent_role_content]

    @classmethod
    def clear_request_cache(cls) -> None:
        """显式清理当前请求线程上的缓存条目；测试或长期复用线程时使用。"""
        try:
            if hasattr(request_local, _REQUEST_CACHE_ATTR):
                delattr(request_local, _REQUEST_CACHE_ATTR)
        except Exception:
            logger.debug("clear_request_cache: request_local 不可用，忽略", exc_info=True)

    @staticmethod
    def _get_request_cache() -> Optional[dict]:
        """惰性获取请求级缓存字典；脱离请求上下文时返回 ``None``，调用方自行降级为不缓存。"""
        try:
            cache = getattr(request_local, _REQUEST_CACHE_ATTR, None)
            if cache is None:
                cache = {}
                setattr(request_local, _REQUEST_CACHE_ATTR, cache)
            return cache
        except Exception:
            logger.debug("request_local 不可用，agent config 缓存降级为不缓存", exc_info=True)
            return None
