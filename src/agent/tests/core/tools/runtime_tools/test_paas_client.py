# -*- coding: utf-8 -*-
"""Test module for PaasSandboxBackend HTTP methods.

通过 mock paas_client 来验证：
- 各 API 方法（create_sandbox、exec_command、upload_file 等）
- 错误处理

注意：该测试不依赖真实的 PaaS 网络环境。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from aidev_agent.core.tools.runtime_tools.paas_backend import (
    PaasSandboxBackend,
)


def _make_response(*, status_code: int = 200, json_data=None, content: bytes = b""):
    """创建一个可用于 mock 的 Response 替身。"""

    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    if status_code >= 400:
        from requests.exceptions import HTTPError

        resp.raise_for_status.side_effect = HTTPError(response=resp)
    return resp


def _make_backend(**kwargs) -> PaasSandboxBackend:
    """创建一个带默认参数的 PaasSandboxBackend 实例，用于测试 HTTP 方法。"""
    defaults = dict(
        app_code="test-app",
        access_token="test-token",
        bk_username="test-username",
        snapshot="",
        snapshot_entrypoint=[],
        env_vars={},
    )
    defaults.update(kwargs)
    return PaasSandboxBackend(**defaults)


class TestPaasSandboxBackendAuth:
    """测试配置项显式注入。"""

    def test_explicit_params(self):
        """所有配置项均通过构造函数显式注入，不依赖任何环境变量。"""
        backend = _make_backend(
            app_code="explicit-app",
            access_token="explicit-token",
        )
        assert backend._app_code == "explicit-app"
        assert backend._access_token == "explicit-token"


class TestPaasSandboxBackendErrors:
    """测试错误处理。"""

    def test_missing_app_code(self):
        backend = _make_backend(app_code="")
        with pytest.raises(ValueError, match="app_code"):
            backend.create_sandbox()


class TestPaasSandboxBackendHTTPMethods:
    """测试各 HTTP API 方法（通过 mock client）。"""

    @pytest.fixture()
    def backend(self):
        b = _make_backend()
        b.client = MagicMock()
        return b

    def test_create_sandbox(self, backend):
        backend.client.create_sandbox.request.return_value = _make_response(json_data={"uuid": "sb-123"})

        sandbox_id = backend.create_sandbox(name="test")
        assert sandbox_id == "sb-123"

        backend.client.create_sandbox.request.assert_called_once_with(
            json={"name": "test"},
            path_params={"app_code": "test-app"},
        )

    def test_destroy_sandbox(self, backend):
        backend.client.delete_sandbox.request.return_value = _make_response()

        backend.destroy_sandbox("sb-123")

        backend.client.delete_sandbox.request.assert_called_once_with(path_params={"sandbox_id": "sb-123"})

    def test_exec_command(self, backend):
        backend.client.exec_command.request.return_value = _make_response(
            json_data={"stdout": "hello\n", "stderr": "", "exit_code": 0}
        )

        result = backend.exec_command("sb-123", "echo hello")
        assert result.stdout == "hello\n"
        assert result.stderr == ""
        assert result.exit_code == 0

        backend.client.exec_command.request.assert_called_once_with(
            json={"cmd": "echo hello"},
            path_params={"sandbox_id": "sb-123"},
        )

    def test_upload_file(self, backend):
        backend.client.upload_file.request.return_value = _make_response(
            json_data={"code": 0, "data": None, "message": "ok"}
        )

        backend.upload_file("sb-123", "/app/test.txt", b"hello content")

        backend.client.upload_file.request.assert_called_once_with(
            files={"file": ("test.txt", b"hello content"), "path": (None, "/app/test.txt")},
            path_params={"sandbox_id": "sb-123"},
        )

    def test_download_file(self, backend):
        content = b"file content here"
        backend.client.download_file.request.return_value = _make_response(content=content)

        result = backend.download_file("sb-123", "/app/test.txt")
        assert result == content

        backend.client.download_file.request.assert_called_once_with(
            params={"path": "/app/test.txt"},
            path_params={"sandbox_id": "sb-123"},
        )

    def test_delete_file(self, backend):
        backend.client.exec_command.request.return_value = _make_response(
            json_data={"stdout": "", "stderr": "", "exit_code": 0}
        )

        backend.delete_file("sb-123", "/app/test.txt", recursive=True)

        backend.client.exec_command.request.assert_called_once_with(
            json={"cmd": "rm -rf /app/test.txt"},
            path_params={"sandbox_id": "sb-123"},
        )
