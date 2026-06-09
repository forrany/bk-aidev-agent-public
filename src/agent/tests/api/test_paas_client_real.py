# -*- coding: utf-8 -*-
"""PaaS Client 真实接口测试（stag 环境）。

需要以下环境变量才能执行真实接口测试：
- BKPAAS_APP_ID: 应用编码
- BKPAAS_APP_SECRET: 应用密钥
- ACCESS_TOKEN: 用户访问令牌

缺少任一环境变量时，所有测试将被 skipif 自动跳过。
"""

import os
import time

import pytest
from aidev_agent.api.paas_client import BkPaaSSandboxApi
from aidev_agent.api.utils import get_endpoint

APP_CODE = os.getenv("BKPAAS_APP_ID", "")
SECRET_KEY = os.getenv("BKPAAS_APP_SECRET", "")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")
BK_API_URL_TMPL = os.getenv("BK_API_URL_TMPL", "")

# 沙箱快照镜像（需要包含 pre_start.sh 的正式镜像，不能用 python:3.11 等通用镜像）
SANDBOX_SNAPSHOT = os.getenv("SANDBOX_SNAPSHOT", "")

skip_no_env = not all([APP_CODE, SECRET_KEY, ACCESS_TOKEN])


@pytest.fixture(scope="module")
def client():
    """创建 PaaS Client。联调阶段 endpoint 指向 stag 环境，上线后改为 prod。"""
    _client = BkPaaSSandboxApi.get_client(app_code=APP_CODE, app_secret=SECRET_KEY)
    # 联调阶段：强制覆盖 endpoint 为 stag 环境（默认为 prod）
    # 上线后移除此行，使用默认 prod endpoint
    _client._endpoint = get_endpoint(BkPaaSSandboxApi._api_name, stage="stag")
    # 设置 access_token 用于用户认证
    _client.update_bkapi_authorization(access_token=ACCESS_TOKEN)
    return _client


@pytest.mark.skipif(
    skip_no_env, reason="缺少必要环境变量 (BKPAAS_APP_ID, BKPAAS_APP_SECRET, ACCESS_TOKEN)，跳过真实接口测试"
)
class TestPaasClientVolumes:
    """PV 接口测试：Volume 增删查与跨沙箱数据共享。"""

    def test_volume_lifecycle(self, client):
        """Volume 增删查生命周期测试。"""
        vol_name = f"test-vol-{int(time.time())}"
        volume_id = None
        try:
            # 创建 Volume
            response = client.create_agent_sandbox_volume.request(
                json={"name": vol_name, "display_name": "测试存储卷"},
                path_params={"app_code": APP_CODE},
            )
            assert response.status_code == 201
            data = response.json()
            volume_id = data["uuid"]
            assert data["name"].startswith("test-vol-")

            # 查询 Volume 列表
            response = client.list_agent_sandbox_volumes.request(
                path_params={"app_code": APP_CODE},
            )
            assert response.status_code == 200
            volumes = response.json()
            assert any(v["uuid"] == volume_id for v in volumes)

            # 删除 Volume
            response = client.delete_agent_sandbox_volume.request(
                path_params={"app_code": APP_CODE, "volume_id": volume_id},
            )
            assert response.status_code == 204
            volume_id = None  # 已删除，无需在 finally 中再次删除
        finally:
            if volume_id is not None:
                client.delete_agent_sandbox_volume.request(
                    path_params={"app_code": APP_CODE, "volume_id": volume_id},
                )

    def test_sandbox_with_volume(self, client):
        """验证 Volume 在两个沙箱间共享数据。"""
        vol_name = f"test-share-vol-{int(time.time())}"
        volume_id = None
        sandbox1_id = None
        sandbox2_id = None
        try:
            # 创建 Volume
            response = client.create_agent_sandbox_volume.request(
                json={"name": vol_name, "display_name": "共享存储卷测试"},
                path_params={"app_code": APP_CODE},
            )
            assert response.status_code == 201
            volume_id = response.json()["uuid"]

            # 创建沙箱1（挂载 volume）
            response = client.create_sandbox.request(
                json={
                    "snapshot": SANDBOX_SNAPSHOT,
                    "snapshot_entrypoint": [],
                    "env_vars": {
                        "ACCESS_TOKEN": ACCESS_TOKEN,
                        "BK_API_URL_TMPL": BK_API_URL_TMPL,
                    },
                    "volume_mounts": [{"volume_id": volume_id, "mount_path": "/data/shared"}],
                },
                path_params={"app_code": APP_CODE},
            )
            assert response.status_code == 201
            sandbox1_id = response.json()["uuid"]

            # 创建沙箱2（挂载同一个 volume）
            response = client.create_sandbox.request(
                json={
                    "snapshot": SANDBOX_SNAPSHOT,
                    "snapshot_entrypoint": [],
                    "env_vars": {
                        "ACCESS_TOKEN": ACCESS_TOKEN,
                        "BK_API_URL_TMPL": BK_API_URL_TMPL,
                    },
                    "volume_mounts": [{"volume_id": volume_id, "mount_path": "/data/shared"}],
                },
                path_params={"app_code": APP_CODE},
            )
            assert response.status_code == 201
            sandbox2_id = response.json()["uuid"]

            # 在沙箱1写入文件
            response = client.exec_command.request(
                json={"cmd": "sh -c \"echo 'hello from sandbox1' > /data/shared/test_cross.txt\""},
                path_params={"sandbox_id": sandbox1_id},
            )
            assert response.status_code == 200

            # 在沙箱2读取文件
            response = client.exec_command.request(
                json={"cmd": "cat /data/shared/test_cross.txt"},
                path_params={"sandbox_id": sandbox2_id},
            )
            assert response.status_code == 200
            assert "hello from sandbox1" in response.json()["stdout"]
        finally:
            # 清理：删除沙箱和 Volume
            if sandbox1_id is not None:
                client.delete_sandbox.request(path_params={"sandbox_id": sandbox1_id})
            if sandbox2_id is not None:
                client.delete_sandbox.request(path_params={"sandbox_id": sandbox2_id})
            if volume_id is not None:
                client.delete_agent_sandbox_volume.request(
                    path_params={"app_code": APP_CODE, "volume_id": volume_id},
                )


@pytest.mark.skipif(
    skip_no_env, reason="缺少必要环境变量 (BKPAAS_APP_ID, BKPAAS_APP_SECRET, ACCESS_TOKEN)，跳过真实接口测试"
)
class TestPaasClientSandbox:
    """已有方法测试：Sandbox 完整生命周期。"""

    def test_sandbox_lifecycle(self, client):
        """完整沙箱生命周期测试：创建 → 执行命令 → 上传文件 → 下载文件 → 删除。"""
        sandbox_id = None
        try:
            # 创建 Sandbox
            response = client.create_sandbox.request(
                json={
                    "snapshot": SANDBOX_SNAPSHOT,
                    "snapshot_entrypoint": [],
                    "env_vars": {
                        "ACCESS_TOKEN": ACCESS_TOKEN,
                        "BK_API_URL_TMPL": BK_API_URL_TMPL,
                    },
                },
                path_params={"app_code": APP_CODE},
            )
            assert response.status_code == 201
            sandbox_id = response.json()["uuid"]

            # 执行命令
            response = client.exec_command.request(
                json={"cmd": "echo hello"},
                path_params={"sandbox_id": sandbox_id},
            )
            assert response.status_code == 200
            assert "hello" in response.json()["stdout"]

            # 上传文件
            response = client.upload_file.request(
                files={"file": ("test.txt", b"hello world"), "path": (None, "/tmp/test.txt")},
                path_params={"sandbox_id": sandbox_id},
            )
            assert response.status_code in (200, 201, 204)

            # 下载文件
            response = client.download_file.request(
                params={"path": "/tmp/test.txt"},
                path_params={"sandbox_id": sandbox_id},
            )
            assert response.content == b"hello world"

            # 删除 Sandbox
            response = client.delete_sandbox.request(
                path_params={"sandbox_id": sandbox_id},
            )
            response.raise_for_status()
            sandbox_id = None  # 已删除，无需在 finally 中再次删除
        finally:
            if sandbox_id is not None:
                client.delete_sandbox.request(path_params={"sandbox_id": sandbox_id})
