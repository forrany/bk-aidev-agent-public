import os
import json
from logging import getLogger
from os import environ
from typing import Tuple

from aidev_agent.api.bk_aidev import BKAidevApi
from django.conf import settings

logger = getLogger(__name__)

# 全局 API client：使用 BKPAAS_APP_ID + BKPAAS_APP_SECRET 认证
bkaidev_api_client = BKAidevApi.get_client(app_code=settings.APP_CODE, app_secret=settings.SECRET_KEY)


def get_flow_agent_client(username: str) -> Tuple:
    """获取 Flow Agent 专用 client 和已构建好的认证请求头。

    注意：平台侧已支持 X-BKAIDEV-USER 认证方式，不再强制要求 access_token。

    Returns:
        (client, headers) 元组：
        - client: BKAidevApi client
        - headers: 包含 X-BKAIDEV-USER 的请求头字典
    """
    flow_client = BKAidevApi.get_client(app_code=settings.APP_CODE, app_secret=settings.SECRET_KEY)
    headers = {"X-BKAIDEV-USER": username}
    return flow_client, headers


def set_user_access_token(request):
    try:
        import bkoauth

        bkoauth.get_access_token(request)
    except Exception as err:
        logger.warning(f"failed to import bkoauth, error: {err}")


def is_local_dev():
    return os.getenv("BKPAAS_ENVIRONMENT", "dev").lower() in {"dev", "development"}
