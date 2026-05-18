import os
from logging import getLogger

logger = getLogger(__name__)


def set_user_access_token(request):
    try:
        import bkoauth

        bkoauth.get_access_token(request)
    except Exception as err:
        logger.warning(f"failed to import bkoauth, error: {err}")


def is_local_dev():
    return os.getenv("BKPAAS_ENVIRONMENT", "dev").lower() in {"dev", "development"}
