# -*- coding: utf-8 -*-
"""Test module for PaasSandboxBackend.

测试范围：
- _build_volume_mounts 方法从 state 中构造 volume_mounts 参数
- __del__ 方法已移除
- 上下文管理器协议（__enter__/__exit__）
- destroy_sandbox timeout 参数传递
- kill() 尽力清理（_sandbox_id 始终置 None）
- kill() 无沙箱时提前返回
- __exit__ 异常时仍调用 kill()
- close() 委托给 kill()

注意：该测试不依赖真实的 PaaS 网络环境。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from aidev_agent.core.tools.runtime_tools.paas_backend import PaasSandboxBackend

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_backend(**kwargs) -> PaasSandboxBackend:
    """创建一个带默认参数的 PaasSandboxBackend 实例。"""
    defaults = dict(
        app_code="test-app",
        bk_username="test-username",
        client=MagicMock(),
        snapshot="",
        snapshot_entrypoint=[],
        env_vars={"STORAGE_PATH": "/app/storage"},
    )
    defaults.update(kwargs)
    return PaasSandboxBackend(**defaults)


# ---------------------------------------------------------------------------
# _build_volume_mounts 测试
# ---------------------------------------------------------------------------


def test_build_volume_mounts_no_pv():
    """state 为 None 或 runtime_paas_sbx_pv 为空时，返回 None。"""
    backend = _make_backend()

    # state 为 None
    assert backend._build_volume_mounts(None) is None

    # 空列表
    assert backend._build_volume_mounts({"runtime_paas_sbx_pv": []}) is None

    # 缺少 key
    assert backend._build_volume_mounts({}) is None


def test_build_volume_mounts_with_pv():
    """state 包含 PV 信息时，返回正确的 volume_mounts 列表。"""
    backend = _make_backend()

    state = {
        "runtime_paas_sbx_pv": [
            {
                "type": "paas-sbx-pv",
                "volume_id": "vol-uuid-123",
                "volume_name": "agent-pv-thread-1",
                "mount_path": "session",
            }
        ]
    }

    result = backend._build_volume_mounts(state)
    assert result is not None
    assert len(result) == 1
    assert result[0]["volume_id"] == "vol-uuid-123"
    assert result[0]["mount_path"] == "/app/storage/session"


def test_build_volume_mounts_ignores_pv_source_field():
    """source 等 state-only 字段不应透传到 PaaS volume_mounts。"""
    backend = _make_backend()

    state = {
        "runtime_paas_sbx_pv": [
            {
                "type": "paas-sbx-pv",
                "volume_id": "vol-uuid-123",
                "volume_name": "agent-pv-thread-1",
                "mount_path": "session",
                "source": "platform",
            },
            {
                "type": "paas-sbx-pv",
                "volume_id": "vol-uuid-456",
                "volume_name": "agent-pv-thread-2",
                "mount_path": "session",
                "source": "runtime",
            },
        ]
    }

    result = backend._build_volume_mounts(state)
    assert result == [
        {"volume_id": "vol-uuid-123", "mount_path": "/app/storage/session"},
        {"volume_id": "vol-uuid-456", "mount_path": "/app/storage/session"},
    ]
    for mount in result:
        assert set(mount) == {"volume_id", "mount_path"}
        assert "source" not in mount
        assert "volume_name" not in mount


def test_build_volume_mounts_storage_path_trailing_slash():
    """STORAGE_PATH 以 / 结尾时，mount_path 不含双斜杠。"""
    backend = _make_backend(env_vars={"STORAGE_PATH": "/app/storage/"})

    state = {
        "runtime_paas_sbx_pv": [
            {
                "type": "paas-sbx-pv",
                "volume_id": "vol-uuid-456",
                "volume_name": "agent-pv-thread-2",
                "mount_path": "session",
            }
        ]
    }

    result = backend._build_volume_mounts(state)
    assert result is not None
    assert len(result) == 1
    # 不应包含双斜杠
    assert result[0]["mount_path"] == "/app/storage/session"
    assert "//" not in result[0]["mount_path"]


# ---------------------------------------------------------------------------
# 清理行为测试
# ---------------------------------------------------------------------------


class TestNoDelMethod:
    """验证 __del__ 方法已移除。"""

    def test_no_del_method(self):
        assert hasattr(PaasSandboxBackend, "__del__") is False


class TestContextManager:
    """验证上下文管理器协议。"""

    def test_context_manager_calls_kill(self):
        """with 语句退出时调用 kill()。"""
        backend = _make_backend()
        backend._sandbox_id = "sb-test"

        with patch.object(backend, "kill") as mock_kill:
            with backend:
                pass
            mock_kill.assert_called_once()

    def test_enter_returns_self(self):
        """__enter__ 返回 self。"""
        backend = _make_backend()
        assert backend.__enter__() is backend

    def test_exit_calls_kill_on_exception(self):
        """__exit__ 无论异常类型如何都调用 kill()。"""
        backend = _make_backend()
        backend._sandbox_id = "sb-test"

        with patch.object(backend, "kill") as mock_kill:
            with pytest.raises(ValueError), backend:
                raise ValueError("test error")
            mock_kill.assert_called_once()


class TestDestroySandboxTimeout:
    """验证 destroy_sandbox timeout 参数传递。"""

    def test_destroy_sandbox_timeout(self):
        """destroy_sandbox 将 timeout 参数传递给 HTTP 请求。"""
        backend = _make_backend()
        backend.client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        backend.client.delete_sandbox.request.return_value = mock_response

        backend.destroy_sandbox("sb-123")

        backend.client.delete_sandbox.request.assert_called_once_with(path_params={"sandbox_id": "sb-123"}, timeout=10)

    def test_destroy_sandbox_custom_timeout(self):
        """destroy_sandbox 支持自定义 timeout。"""
        backend = _make_backend()
        backend.client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        backend.client.delete_sandbox.request.return_value = mock_response

        backend.destroy_sandbox("sb-123", timeout=30)

        backend.client.delete_sandbox.request.assert_called_once_with(path_params={"sandbox_id": "sb-123"}, timeout=30)


class TestKillBehavior:
    """验证 kill() 行为。"""

    def test_kill_sets_sandbox_id_none_on_failure(self):
        """kill() 在 destroy_sandbox 失败后仍将 _sandbox_id 设为 None（尽力清理）。"""
        backend = _make_backend()
        backend._sandbox_id = "sb-fail"

        with patch.object(
            backend,
            "destroy_sandbox",
            side_effect=Exception("permanent failure"),
        ):
            backend.kill()

        assert backend._sandbox_id is None

    def test_kill_returns_early_when_no_sandbox(self):
        """kill() 在 _sandbox_id 为 None 时立即返回，无 HTTP 调用。"""
        backend = _make_backend()
        assert backend._sandbox_id is None

        with patch.object(backend, "destroy_sandbox") as mock_destroy:
            backend.kill()

        mock_destroy.assert_not_called()

    def test_kill_succeeds(self):
        """kill() 成功时将 _sandbox_id 设为 None。"""
        backend = _make_backend()
        backend._sandbox_id = "sb-ok"

        with patch.object(backend, "destroy_sandbox"):
            backend.kill()

        assert backend._sandbox_id is None


class TestCloseDelegatesToKill:
    """验证 close() 委托给 kill()。"""

    def test_close_calls_kill(self):
        backend = _make_backend()
        backend.kill = MagicMock()
        backend.close()
        backend.kill.assert_called_once()
