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
    from aidev_agent.packages.opentelemetry import BkAidevAgentInstrumentor
    from aidev_agent.packages.opentelemetry.config import OTelConfig

    from aidev_bkplugin.utils import get_otel_endpoints
except ImportError:
    BkAidevAgentInstrumentor = None
    OTelConfig = None
    get_otel_endpoints = None

try:
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
except ImportError:
    HTTPXClientInstrumentor = None

try:
    from opentelemetry.instrumentation.threading import ThreadingInstrumentor
except ImportError:
    ThreadingInstrumentor = None

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
            otel_config = OTelConfig(otel_endpoints=get_otel_endpoints())
            BkAidevAgentInstrumentor(config=otel_config).instrument()
        else:
            logger.info(
                "[aidev_bkplugin] OpenTelemetry extras 未安装，跳过自动 instrument；"
                "如需启用请安装 aidev-bkplugin[opentelemetry]。"
            )
        # 注入 httpx (LLM 网关底层 HTTP 客户端) 的 trace 传播
        # 使 LLM 调用自动携带 traceparent header，网关可加入分布式追踪
        try:
            HTTPXClientInstrumentor().instrument()
        except Exception:  # noqa: BLE001
            logger.debug("opentelemetry-instrumentation-httpx not available, skipping httpx instrumentation")
        # 注入跨线程的 trace 传递，因为知识库中 LLM 提交是异步的
        try:
            ThreadingInstrumentor().instrument()
        except Exception:  # noqa: BLE001
            logger.debug("opentelemetry-instrumentation-threading not available, skipping httpx instrumentation")

        return super().ready()
