# -*- coding: utf-8 -*-

from aidev_agent.api.utils import get_endpoint
from aidev_agent.config import settings

# 网关接口
BKAIDEV_URL = get_endpoint(settings.BK_AIDEV_GATEWAY_NAME, settings.BK_APIGW_STAGE)
