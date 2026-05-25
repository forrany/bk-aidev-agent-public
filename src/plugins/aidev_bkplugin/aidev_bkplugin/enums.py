# -*- coding: utf-8 -*-

import enum


class PluginPollTaskState(enum.Enum):
    """``poll_task_state`` 返回的任务轮询状态。"""

    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
