# -*- coding: utf-8 -*-
import json
from json.decoder import JSONDecodeError

from blueapps.core.exceptions import BlueException, ServerBlueException
from blueapps.utils.logger import logger
from rest_framework import status
from rest_framework.views import exception_handler


class InternalServerError(ServerBlueException):
    """
    内部服务器错误：用于捕获未处理的系统异常（如 TypeError、ValueError 等）
    区别于一般的 ServerBlueException，使用专属错误码 533 便于问题定位
    """

    MESSAGE = "内部服务器错误"
    ERROR_CODE = "533"


def custom_exception_handler(exc, context):
    """
    分类：
        APIException及子类异常
        app自定义异常和未处理异常
    """
    response = exception_handler(exc, context)
    if response:
        return response

    # response
    exc_data = None
    exc_message = str(exc)
    code = status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    if getattr(exc, "response", None) is not None:
        exc_data = exc.response.json()
        if exc_data.get("error") and isinstance(exc_data["error"], dict):
            code = exc_data["error"].get("code")
            exc_message = exc_data["error"].get("message")
            exc_data = exc_data["error"].get("data")
        else:
            exc_message = exc_data.get("message") or exc_data.get("detail") or exc.response.text
            code = exc_data.get("code") or code
    else:
        if hasattr(exc, "data") and exc.data:
            if not isinstance(exc.data, str):
                exc_data = exc.data
            try:
                exc_data = json.loads(exc.data)
            except JSONDecodeError:
                exc_data = exc.data
            except TypeError:
                # 其它内容不能被json解析 忽略
                pass

        if hasattr(exc, "message") and exc.message:
            try:
                exc_message = json.loads(str(exc.message))
            except JSONDecodeError:
                exc_message = str(exc.message)

        if isinstance(exc, BlueException):
            code = exc.code
            status_code = exc.STATUS_CODE

    # 使用json方便提取
    logger.exception(json.dumps({"code": code, "message": exc_message, "status_code": status_code, "data": exc_data}))
    # 对于 BlueException 类型的异常，保持原有行为
    if isinstance(exc, BlueException):
        raise BlueException(message=exc_message, code=code, status_code=status_code, data=exc_data)
    else:
        # 对于非 BlueException 的异常（如 TypeError、ValueError 等系统异常），
        # 使用 InternalServerError 表示内部服务器错误（错误码 533），便于问题定位
        raise InternalServerError(message=exc_message, data=exc_data)
