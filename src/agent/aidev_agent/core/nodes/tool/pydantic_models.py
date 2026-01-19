# -*- coding: utf-8 -*-

from pydantic import BaseModel


class ToolNodeSettings(BaseModel):
    """ToolNode wrappers settings."""

    use_timer: bool = True
    use_result_limit: bool = False
    result_limit_thrd: int = 1000
