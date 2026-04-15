# -*- coding: utf-8 -*-
"""aidev_agent.core.tools.runtime_tools.types

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

运行时后端共享数据类型定义。

该模块包含所有运行时后端（local/e2b/paas）共享的类型合约：
- ls/glob 返回的 FileInfo
- grep 返回的 GrepMatch
- write/edit/execute 返回的结果结构
- upload/download 返回的结构

注意：该模块不包含任何具体后端实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, Protocol

from typing_extensions import TypedDict


class FileInfo(TypedDict):
    """文件信息结构。

    用于 ls_info 和 glob_info 方法返回的文件元数据。
    只有 path 是必需的，其他字段根据后端能力可选提供。
    """

    path: str
    """文件或目录的路径"""

    is_dir: NotRequired[bool]
    """是否为目录"""

    size: NotRequired[int]
    """文件大小（字节）"""

    modified_at: NotRequired[str]
    """最后修改时间（ISO 8601 格式）"""


class GrepMatch(TypedDict):
    """grep 搜索匹配结果结构。"""

    path: str
    """匹配的文件路径"""

    line: int
    """匹配的行号"""

    text: str
    """匹配的行内容"""


@dataclass
class WriteResult:
    """写操作结果。

    Attributes:
        error: 失败时的错误信息，成功时为 None
        path: 写入的文件路径，失败时为 None
        files_update: 文件更新信息，外部存储时为 None
    """

    error: str | None = None
    path: str | None = None
    files_update: dict | None = None


@dataclass
class EditResult:
    """编辑操作结果。

    Attributes:
        error: 失败时的错误信息，成功时为 None
        path: 编辑的文件路径，失败时为 None
        occurrences: 替换的匹配数量，失败时为 None
        files_update: 文件更新信息，外部存储时为 None
    """

    error: str | None = None
    path: str | None = None
    occurrences: int | None = None
    files_update: dict | None = None


@dataclass
class ExecuteResult:
    """命令执行结果。

    Attributes:
        output: 命令的标准输出和标准错误的合并输出
        exit_code: 命令退出码，None 表示执行过程中发生错误
        truncated: 输出是否因大小限制被截断
    """

    output: str
    exit_code: int | None = None
    truncated: bool = False


class FileUploadResponse(TypedDict):
    """文件上传响应结构。"""

    path: str
    """上传的文件路径"""

    error: str | None
    """错误信息，成功时为 None"""


class FileDownloadResponse(TypedDict):
    """文件下载响应结构。"""

    path: str
    """下载的文件路径"""

    content: bytes | None
    """文件内容，失败时为 None"""

    error: str | None
    """错误信息，成功时为 None"""


class RuntimeBackend(Protocol):
    """运行时后端抽象（基于 tool_provider 的实际使用定义）。

    说明：该 Protocol 仅描述 runtime tool provider 调用到的最小方法集合。
    具体后端实现可提供更多能力，但至少需要满足这些方法签名。
    """

    def ls_info(self, path: str) -> list[FileInfo]: ...

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str: ...

    def write(self, file_path: str, content: str) -> WriteResult: ...

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult: ...

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]: ...

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str: ...

    def execute(self, command: str, timeout: int = 120, max_output_size: int = 100000) -> ExecuteResult: ...

    async def aexecute(self, command: str, timeout: int = 120, max_output_size: int = 100000) -> ExecuteResult: ...


__all__ = [
    "FileInfo",
    "GrepMatch",
    "WriteResult",
    "EditResult",
    "ExecuteResult",
    "FileUploadResponse",
    "FileDownloadResponse",
    "RuntimeBackend",
]
