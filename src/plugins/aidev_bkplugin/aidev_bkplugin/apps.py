# -*- coding: utf-8 -*-

import logging

from aidev_agent.utils.module_loading import import_string
from django.apps import AppConfig
from django.conf import settings

try:
    import bkoauth
except ImportError:
    bkoauth = None

# OpenTelemetry 是可选 extras，未安装时降级为 no-op。
# 安装方式：pip install aidev-bkplugin[opentelemetry]
try:
    from aidev_bkplugin.packages.opentelemetry import BkAidevAgentInstrumentor
except ImportError:
    BkAidevAgentInstrumentor = None

logger = logging.getLogger(__name__)


class AgentConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "aidev_bkplugin"

    def ready(self) -> None:
        from aidev_agent.packages.resource_manager import resource_manager
        from aidev_agent.services.common_agent import CommonQAAgent, common_agent_factory

        if bkoauth:
            bkoauth._init_function()

        # 仅当 AIDEV_AGENT 解析出的类与 SDK 默认 CommonQAAgent 不一致时，实例化后注入；
        # 缺省值即继续使用 SDK 默认实例（来自 common_agent_factory 的 default）。
        custom_agent_cls = import_string(settings.AIDEV_AGENT)
        if custom_agent_cls is not CommonQAAgent:
            common_agent_factory.replace_defaults(custom_agent_cls())

        custom_resource_manager = getattr(settings, "AIDEV_RESOURCE_MANAGER", "")
        if custom_resource_manager:
            resource_manager.replace_defaults(import_string(custom_resource_manager)())

        if BkAidevAgentInstrumentor is not None:
            BkAidevAgentInstrumentor().instrument()
        else:
            logger.info(
                "[aidev_bkplugin] OpenTelemetry extras 未安装，跳过自动 instrument；"
                "如需启用请安装 aidev-bkplugin[opentelemetry]。"
            )

        return super().ready()
