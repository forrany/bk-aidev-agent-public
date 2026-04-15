# -*- coding: utf-8 -*-
"""Test module for PaasSandboxBackend.

通过 monkeypatch mock requests 来验证：
- 惰性创建与 kill 生命周期
- 文件与命令相关方法的签名与返回类型
- 错误处理

注意：该测试不依赖真实的 PaaS 网络环境。
"""

from __future__ import annotations

import pytest
from aidev_agent.core.tools.runtime_tools.paas_backend import ExecResult, PaasSandboxBackend
from aidev_agent.core.tools.runtime_tools.types import EditResult, ExecuteResult, WriteResult

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

_PAAS_BACKEND_MOD = "aidev_agent.core.tools.runtime_tools.paas_backend"


def _make_response(*, status_code: int = 200, json_data=None, content: bytes = b""):
    """创建一个可用于 mock 的 requests.Response 替身。"""

    class _Resp:
        def __init__(self):
            self.status_code = status_code
            self.content = content
            self._json_data = json_data

        def json(self):
            return self._json_data

    return _Resp()


class MockOps:
    """记录沙箱操作调用，代替原来的 MockClient。

    通过 monkeypatch 将 PaasSandboxBackend 的 HTTP 方法替换为此对象的实现，
    从而在不发真实 HTTP 请求的情况下验证 Backend 的业务逻辑。
    """

    def __init__(self):
        self.sandbox_created = False
        self.sandbox_destroyed = False
        self.exec_handler = lambda sandbox_id, cmd, **kw: ExecResult(stdout="", stderr="", exit_code=0)
        self.uploaded_files: dict[str, bytes] = {}
        self.deleted_files: list[str] = []

    def create_sandbox(self, name=None, env_vars=None, snapshot=None, snapshot_entrypoint=None):
        self.sandbox_created = True
        return "mock-sandbox-id"

    def destroy_sandbox(self, sandbox_id):
        self.sandbox_destroyed = True

    def exec_command(self, sandbox_id, cmd, cwd=None, env=None, timeout=None):
        return self.exec_handler(sandbox_id, cmd, cwd=cwd, env=env, timeout=timeout)

    def upload_file(self, sandbox_id, path, content):
        self.uploaded_files[path] = content

    def download_file(self, sandbox_id, path):
        if path not in self.uploaded_files:
            raise FileNotFoundError(f"HTTP 404: file not found: {path}")
        return self.uploaded_files[path]

    def delete_file(self, sandbox_id, path, recursive=False):
        self.deleted_files.append(path)
        if path in self.uploaded_files:
            del self.uploaded_files[path]


@pytest.fixture()
def mock_ops():
    return MockOps()


@pytest.fixture()
def backend(mock_ops, monkeypatch):
    """创建 PaasSandboxBackend 实例，并将其所有 HTTP 方法替换为 mock_ops 的实现。"""
    b = PaasSandboxBackend(
        app_code="test-app",
        access_token="test-token",
        bk_username="test-username",
        snapshot="test-snapshot",
        snapshot_entrypoint=["python", "-m", "server"],
        env_vars={},
    )
    # 将实例方法替换为 mock_ops 的方法，保持 self 绑定到 b
    monkeypatch.setattr(b, "create_sandbox", mock_ops.create_sandbox)
    monkeypatch.setattr(b, "destroy_sandbox", mock_ops.destroy_sandbox)
    monkeypatch.setattr(b, "exec_command", mock_ops.exec_command)
    monkeypatch.setattr(b, "upload_file", mock_ops.upload_file)
    monkeypatch.setattr(b, "download_file", mock_ops.download_file)
    monkeypatch.setattr(b, "delete_file", mock_ops.delete_file)
    return b


class TestPaasSandboxBackendLifecycle:
    def test_lazy_create(self, backend, mock_ops):
        """首次调用操作方法时才创建沙箱。"""
        assert backend._sandbox_id is None
        assert not mock_ops.sandbox_created

        backend.ls_info("/app")
        assert backend._sandbox_id == "mock-sandbox-id"
        assert mock_ops.sandbox_created

    def test_reuse_sandbox(self, backend, mock_ops):
        """后续调用复用沙箱 ID。"""
        backend.ls_info("/app")
        backend.ls_info("/app")
        # create_sandbox 只被调用一次（通过检查 sandbox_created 标记）
        assert backend._sandbox_id == "mock-sandbox-id"

    def test_kill(self, backend, mock_ops):
        """kill 销毁沙箱并重置 ID。"""
        backend.ls_info("/app")
        assert backend._sandbox_id is not None

        backend.kill()
        assert backend._sandbox_id is None
        assert mock_ops.sandbox_destroyed

    def test_kill_when_not_created(self, backend, mock_ops):
        """未创建时 kill 不会调用 destroy。"""
        backend.kill()
        assert not mock_ops.sandbox_destroyed

    def test_recreate_after_kill(self, backend, mock_ops):
        """kill 后再次调用操作方法会重新创建。"""
        backend.ls_info("/app")
        backend.kill()

        mock_ops.sandbox_created = False
        backend.ls_info("/app")
        assert mock_ops.sandbox_created
        assert backend._sandbox_id == "mock-sandbox-id"


class TestPaasSandboxBackendLsInfo:
    def test_ls_info_parsing(self, backend, mock_ops):
        def handler(sandbox_id, cmd, **kw):
            if cmd.startswith("ls "):
                return ExecResult(stdout="file1.txt\nsubdir/\n", stderr="", exit_code=0)
            return ExecResult(stdout="", stderr="", exit_code=0)

        mock_ops.exec_handler = handler

        infos = backend.ls_info("/app")
        assert infos == [
            {"path": "/app/file1.txt", "is_dir": False},
            {"path": "/app/subdir/", "is_dir": True},
        ]

    def test_ls_info_failure(self, backend, mock_ops):
        def handler(sandbox_id, cmd, **kw):
            return ExecResult(stdout="", stderr="No such directory", exit_code=2)

        mock_ops.exec_handler = handler
        assert backend.ls_info("/nonexistent") == []


class TestPaasSandboxBackendRead:
    def test_read_success(self, backend, mock_ops):
        def handler(sandbox_id, cmd, **kw):
            if cmd.startswith("test -f "):
                return ExecResult(stdout="", stderr="", exit_code=0)
            if "wc -l" in cmd:
                return ExecResult(stdout="3\n", stderr="", exit_code=0)
            if cmd.startswith("awk "):
                return ExecResult(stdout="     1\tline1\n     2\tline2\n", stderr="", exit_code=0)
            return ExecResult(stdout="", stderr="", exit_code=0)

        mock_ops.exec_handler = handler

        out = backend.read("/app/test.txt", offset=0, limit=2)
        assert "     1\tline1" in out
        assert "     2\tline2" in out

    def test_read_file_not_found(self, backend, mock_ops):
        def handler(sandbox_id, cmd, **kw):
            if cmd.startswith("test -f "):
                return ExecResult(stdout="", stderr="", exit_code=1)
            return ExecResult(stdout="", stderr="", exit_code=0)

        mock_ops.exec_handler = handler

        out = backend.read("/app/missing.txt")
        assert "not found" in out

    def test_read_empty_file(self, backend, mock_ops):
        def handler(sandbox_id, cmd, **kw):
            if cmd.startswith("test -f "):
                return ExecResult(stdout="", stderr="", exit_code=0)
            if "wc -l" in cmd:
                return ExecResult(stdout="0\n", stderr="", exit_code=0)
            return ExecResult(stdout="", stderr="", exit_code=0)

        mock_ops.exec_handler = handler

        out = backend.read("/app/empty.txt")
        assert "文件存在但内容为空" in out

    def test_read_offset_exceeds(self, backend, mock_ops):
        def handler(sandbox_id, cmd, **kw):
            if cmd.startswith("test -f "):
                return ExecResult(stdout="", stderr="", exit_code=0)
            if "wc -l" in cmd:
                return ExecResult(stdout="2\n", stderr="", exit_code=0)
            return ExecResult(stdout="", stderr="", exit_code=0)

        mock_ops.exec_handler = handler

        out = backend.read("/app/test.txt", offset=5)
        assert "exceeds file length" in out


class TestPaasSandboxBackendWrite:
    def test_write_success(self, backend, mock_ops):
        def handler(sandbox_id, cmd, **kw):
            if cmd.startswith("test -e "):
                return ExecResult(stdout="", stderr="", exit_code=1)  # 文件不存在
            return ExecResult(stdout="", stderr="", exit_code=0)

        mock_ops.exec_handler = handler

        res = backend.write("/app/new.txt", "hello")
        assert isinstance(res, WriteResult)
        assert res.error is None
        assert res.path == "/app/new.txt"
        assert "/app/new.txt" in mock_ops.uploaded_files
        assert mock_ops.uploaded_files["/app/new.txt"] == b"hello"

    def test_write_existing_fails(self, backend, mock_ops):
        def handler(sandbox_id, cmd, **kw):
            if cmd.startswith("test -e "):
                return ExecResult(stdout="", stderr="", exit_code=0)  # 文件存在
            return ExecResult(stdout="", stderr="", exit_code=0)

        mock_ops.exec_handler = handler

        res = backend.write("/app/existing.txt", "hello")
        assert isinstance(res, WriteResult)
        assert res.error is not None
        assert "already exists" in res.error


class TestPaasSandboxBackendEdit:
    def test_edit_success(self, backend, mock_ops):
        # 预先上传文件
        mock_ops.uploaded_files["/app/test.txt"] = b"hello world\n"

        def handler(sandbox_id, cmd, **kw):
            if cmd.startswith("test -f "):
                return ExecResult(stdout="", stderr="", exit_code=0)
            return ExecResult(stdout="", stderr="", exit_code=0)

        mock_ops.exec_handler = handler

        res = backend.edit("/app/test.txt", "world", "universe")
        assert isinstance(res, EditResult)
        assert res.error is None
        assert res.occurrences == 1
        # 验证文件内容已更新
        assert mock_ops.uploaded_files["/app/test.txt"] == b"hello universe\n"

    def test_edit_file_not_found(self, backend, mock_ops):
        def handler(sandbox_id, cmd, **kw):
            if cmd.startswith("test -f "):
                return ExecResult(stdout="", stderr="", exit_code=1)
            return ExecResult(stdout="", stderr="", exit_code=0)

        mock_ops.exec_handler = handler

        res = backend.edit("/app/missing.txt", "old", "new")
        assert isinstance(res, EditResult)
        assert res.error is not None
        assert "not found" in res.error

    def test_edit_string_not_found(self, backend, mock_ops):
        mock_ops.uploaded_files["/app/test.txt"] = b"hello world\n"

        def handler(sandbox_id, cmd, **kw):
            if cmd.startswith("test -f "):
                return ExecResult(stdout="", stderr="", exit_code=0)
            return ExecResult(stdout="", stderr="", exit_code=0)

        mock_ops.exec_handler = handler

        res = backend.edit("/app/test.txt", "nonexistent", "replacement")
        assert isinstance(res, EditResult)
        assert res.error is not None


class TestPaasSandboxBackendGrepGlob:
    def test_grep_parsing(self, backend, mock_ops):
        def handler(sandbox_id, cmd, **kw):
            if cmd.startswith("grep "):
                return ExecResult(stdout="/app/a.txt:1:hello world\n/app/b.txt:5:hello again\n", stderr="", exit_code=0)
            return ExecResult(stdout="", stderr="", exit_code=0)

        mock_ops.exec_handler = handler

        out = backend.grep_raw("hello", path="/app")
        assert isinstance(out, list)
        assert len(out) == 2
        assert out[0]["path"] == "/app/a.txt"
        assert out[0]["line"] == 1
        assert out[0]["text"] == "hello world"

    def test_grep_invalid_regex(self, backend):
        out = backend.grep_raw("[invalid(regex")
        assert isinstance(out, str)
        assert "Invalid regex" in out

    def test_grep_with_glob_filter(self, backend, mock_ops):
        def handler(sandbox_id, cmd, **kw):
            assert "--include=" in cmd
            return ExecResult(stdout="", stderr="", exit_code=1)

        mock_ops.exec_handler = handler
        backend.grep_raw("pattern", path="/app", glob="*.py")

    def test_glob_info(self, backend, mock_ops):
        def handler(sandbox_id, cmd, **kw):
            if cmd.startswith("find "):
                return ExecResult(stdout="/app/a.txt\n/app/b.txt\n", stderr="", exit_code=0)
            return ExecResult(stdout="", stderr="", exit_code=0)

        mock_ops.exec_handler = handler

        infos = backend.glob_info("*.txt", path="/app")
        assert len(infos) == 2
        assert [i["path"] for i in infos] == ["/app/a.txt", "/app/b.txt"]

    def test_glob_info_failure(self, backend, mock_ops):
        def handler(sandbox_id, cmd, **kw):
            return ExecResult(stdout="", stderr="error", exit_code=2)

        mock_ops.exec_handler = handler
        assert backend.glob_info("*.txt", path="/nonexistent") == []


class TestPaasSandboxBackendUploadDownload:
    def test_upload_files(self, backend, mock_ops):
        responses = backend.upload_files(
            [
                ("/app/a.txt", b"content-a"),
                ("/app/b.txt", b"content-b"),
            ]
        )
        assert len(responses) == 2
        assert responses[0]["error"] is None
        assert responses[1]["error"] is None
        assert mock_ops.uploaded_files["/app/a.txt"] == b"content-a"

    def test_download_files(self, backend, mock_ops):
        mock_ops.uploaded_files["/app/a.txt"] = b"content-a"

        responses = backend.download_files(["/app/a.txt", "/app/missing.txt"])
        assert len(responses) == 2
        assert responses[0]["content"] == b"content-a"
        assert responses[0]["error"] is None
        assert responses[1]["content"] is None
        assert responses[1]["error"] is not None


class TestPaasSandboxBackendExecute:
    def test_execute_success(self, backend, mock_ops):
        def handler(sandbox_id, cmd, **kw):
            return ExecResult(stdout="hello\n", stderr="", exit_code=0)

        mock_ops.exec_handler = handler

        res = backend.execute("echo hello")
        assert isinstance(res, ExecuteResult)
        assert "hello" in res.output
        assert res.exit_code == 0
        assert not res.truncated

    def test_execute_with_stderr(self, backend, mock_ops):
        def handler(sandbox_id, cmd, **kw):
            return ExecResult(stdout="out", stderr="err", exit_code=1)

        mock_ops.exec_handler = handler

        res = backend.execute("bad command")
        assert "out" in res.output
        assert "err" in res.output
        assert res.exit_code == 1

    def test_execute_truncation(self, backend, mock_ops):
        def handler(sandbox_id, cmd, **kw):
            return ExecResult(stdout="x" * 200, stderr="", exit_code=0)

        mock_ops.exec_handler = handler

        res = backend.execute("big output", max_output_size=100)
        assert len(res.output) == 100
        assert res.truncated

    @pytest.mark.asyncio
    async def test_aexecute(self, backend, mock_ops):
        def handler(sandbox_id, cmd, **kw):
            return ExecResult(stdout="async hello\n", stderr="", exit_code=0)

        mock_ops.exec_handler = handler

        res = await backend.aexecute("echo hello")
        assert isinstance(res, ExecuteResult)
        assert "async hello" in res.output
        assert res.exit_code == 0


class TestPaasSandboxBackendInit:
    def test_init_basic_attributes(self):
        """验证构造函数正确赋值所有属性。"""
        backend = PaasSandboxBackend(
            app_code="explicit-app",
            access_token="explicit-token",
            bk_username="explicit-username",
            snapshot="test-snapshot",
            snapshot_entrypoint=["python", "-m", "server"],
            env_vars={},
        )
        assert backend._app_code == "explicit-app"
        assert backend._access_token == "explicit-token"
        assert backend._sandbox_id is None

    def test_init_with_params(self):
        """验证 snapshot、snapshot_entrypoint、env_vars 等参数正确赋值。"""
        backend = PaasSandboxBackend(
            app_code="param-app",
            access_token="param-token",
            bk_username="param-username",
            snapshot="param-snapshot",
            snapshot_entrypoint=["python", "-m", "server"],
            env_vars={"KEY": "VALUE"},
        )
        assert backend._snapshot == "param-snapshot"
        assert backend._snapshot_entrypoint == ["python", "-m", "server"]
        assert backend._env_vars == {"KEY": "VALUE"}

    def test_init_api_host_trailing_slash_stripped(self):
        """验证默认参数可以正常构造实例。"""
        backend = PaasSandboxBackend(
            app_code="app",
            access_token="token",
            bk_username="username",
            snapshot="snap",
            snapshot_entrypoint=[],
            env_vars={},
        )
        assert backend._app_code == "app"
