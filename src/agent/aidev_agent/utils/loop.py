# -*- coding: utf-8 -*-
"""
TencentBlueKing is pleased to support the open source community by making
蓝鲸智云 - AIDev (BlueKing - AIDev) available.
Copyright (C) 2025 THL A29 Limited,
a Tencent company. All rights reserved.
Licensed under the MIT License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the License for the
specific language governing permissions and limitations under the License.
We undertake not to change the open source license (MIT license) applicable
to the current version of the project delivered to anyone in the future.
"""

import asyncio
import atexit
import contextlib
import threading

# Thread-local storage for event loops
_thread_local = threading.local()


def get_event_loop():
    """
    Get the current event loop for this thread, create one if it doesn't exist.

    Returns:
        asyncio.AbstractEventLoop: The current event loop for this thread.
    """
    # Try to get the running loop first — it takes priority over the cached one
    try:
        running_loop = asyncio.get_running_loop()
        if hasattr(_thread_local, "loop") and _thread_local.loop is not running_loop:
            # Cached loop is stale (e.g. pytest created a new loop), update cache
            _thread_local.loop = running_loop
        return running_loop
    except RuntimeError:
        pass

    # Check if we have a cached loop for this thread
    if hasattr(_thread_local, "loop") and _thread_local.loop is not None and not _thread_local.loop.is_closed():
        return _thread_local.loop

    # No running loop, try to get the event loop for this thread
    try:
        current_loop = asyncio.get_event_loop()
        # Check if this loop is running
        if current_loop.is_running():
            # If the loop is running, we can't use run_until_complete on it
            # Create a new loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            _thread_local.loop = new_loop
            # Register cleanup function for this thread's loop
            atexit.register(_cleanup_thread_loop, new_loop)
            return new_loop
        else:
            _thread_local.loop = current_loop
            return current_loop
    except RuntimeError:
        # No event loop exists, create a new one
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        _thread_local.loop = new_loop
        # Register cleanup function for this thread's loop
        atexit.register(_cleanup_thread_loop, new_loop)
        return new_loop


def close_thread_loop() -> None:
    """Close and clear the cached event loop for the current thread."""
    loop = getattr(_thread_local, "loop", None)
    if loop is None:
        return

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is loop:
        return

    _cleanup_thread_loop(loop)
    with contextlib.suppress(Exception):
        asyncio.set_event_loop(None)
    _thread_local.loop = None


def run_coro_sync(coro, timeout=None):
    """Run a coroutine synchronously in the current thread's event loop.

    Note: The event loop is intentionally kept open after execution, so that
    long-lived async clients (e.g. httpx.AsyncClient) can safely reuse their
    connections across multiple calls on the same thread. Cleanup is handled by
    atexit registration and by async_to_sync_generator's finally block.
    """
    loop = get_event_loop()
    if timeout is not None:
        coro = asyncio.wait_for(coro, timeout=timeout)
    return loop.run_until_complete(coro)


def _cleanup_thread_loop(loop):
    """Cleanup function to properly close a thread's event loop on exit."""
    if loop is not None and not loop.is_closed():
        try:
            # Cancel all pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()

            # Run until all tasks are cancelled if the loop is not running
            if pending and not loop.is_running():
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

            with contextlib.suppress(AttributeError, RuntimeError):
                loop.run_until_complete(loop.shutdown_asyncgens())
            with contextlib.suppress(AttributeError, RuntimeError):
                loop.run_until_complete(loop.shutdown_default_executor())

            # Close the loop
            loop.close()
        except Exception:
            # Ignore exceptions during cleanup
            pass
