# -*- coding: utf-8 -*-
"""aidev_agent.core.tools.runtime_tools.provider

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

运行时中间件与客户端工具构造。

本模块提供：
- `RuntimeBackendResolver`：运行时解析器（注册运行时、按 runtime 解析 backend、提供参数描述）
- `get_xxx_tool(...)`：构造单个工具（ls/read_file/.../execute）
- `get_client_tools(...)`：构造一组客户端工具（固定 7 个）

每个工具都暴露必填的 `target_runtime` 参数，用于指定目标运行时。

注意：参数名不能使用 `runtime`，因为 LangGraph ToolNode 会将名为 `runtime` 的参数
自动识别为注入点并注入 `ToolRuntime` 对象，与我们的字符串类型冲突。

注意：本模块只实现运行时路由与工具构造，不引入新的后端实现。
"""

from __future__ import annotations

import os
import re
from typing import Annotated, Callable, Literal

from langchain_core.tools import BaseTool, StructuredTool

from .types import RuntimeBackend
from .utils import format_grep_matches, truncate_if_too_long

# ========== 默认配置 ==========

DEFAULT_READ_OFFSET = 0
DEFAULT_READ_LIMIT = 100


# ========== 工具描述常量 ==========

LIST_FILES_TOOL_DESCRIPTION = """列出目录中的所有文件。用于浏览文件系统并找到需要读取或编辑的文件。"""

READ_FILE_TOOL_DESCRIPTION = """从文件系统读取文件。
如果用户提供了文件路径，假设该路径是有效的。读取不存在的文件是允许的；工具会返回错误信息。

用法：
- **重要（大文件与代码库探索）**：使用 offset / limit 参数分页，避免上下文溢出
- 首次扫描：read_file(path, limit=100) 查看文件结构
- 继续读取：read_file(path, offset=100, limit=200) 读取后续 200 行
- 仅在确有必要编辑时才省略 limit（读取全文件）
- 显式指定 offset 与 limit：read_file(path, offset=0, limit=100) 读取前 100 行
- 返回结果使用 cat -n 格式，行号从 1 开始
- 单行超过 5,000 字符会被拆分为多行并附带续行标记（如 5.1、5.2）。当你指定 limit 时，这些续行也计入 limit。
- 你可以在一次回复中调用多个工具；通常最好批量读取多个可能有用的文件
- 如果读取到的文件存在但内容为空，会返回 system reminder 提示"""

WRITE_FILE_TOOL_DESCRIPTION = """在文件系统中新建文件并写入内容。
用法：
- write_file 会创建一个新文件
- 如无必要，优先使用 edit_file 编辑已有文件，尽量避免创建新文件
"""

EDIT_FILE_TOOL_DESCRIPTION = """对文件执行精确字符串替换。
用法：
- 编辑前必须先读取文件；若未读取就尝试编辑，工具会报错
- 编辑时请严格保留 read 输出中的缩进（tab/space）。old_string / new_string 中不要包含行号前缀
- 总是优先编辑已有文件，而不是创建新文件
- 仅在用户明确要求时才使用表情符号
"""

GLOB_TOOL_DESCRIPTION = """按 glob 模式查找文件。
支持标准 glob 模式：`*`（任意字符）、`**`（任意目录）、`?`（单个字符）。
返回匹配该模式的绝对路径列表。
示例：
- `**/*.py` - 查找所有 Python 文件
- `*.txt` - 查找根目录下的所有文本文件
- `/subdir/**/*.md` - 查找 /subdir 下的所有 Markdown 文件"""

GREP_TOOL_DESCRIPTION = """在多个文件中搜索文本模式。
按字面文本（非正则）进行搜索，并根据 output_mode 返回匹配的文件列表或内容。
示例：
- 搜索所有文件：`grep(pattern="TODO")`
- 仅搜索 Python 文件：`grep(pattern="import", glob="*.py")`
- 展示匹配行：`grep(pattern="error", output_mode="content")`"""

EXECUTE_TOOL_DESCRIPTION = """执行 shell 命令。
用法：
执行给定命令，并进行必要的处理与安全措施。

在执行命令前，请遵循以下步骤：
1. 目录确认：
   - 如果命令会创建新目录或新文件，请先使用 ls 确认父目录存在且位置正确
   - 例如，在运行 "mkdir foo/bar" 之前，先用 ls 检查 "foo" 是否存在且为目标父目录
2. 命令执行：
   - 对包含空格的路径始终使用双引号（例如：cd "path with spaces/file.txt"）
   - 正确示例：
     - cd "/Users/name/My Documents"
     - python "/path/with spaces/script.py"
   - 错误示例（会失败）：
     - cd /Users/name/My Documents
     - python /path/with spaces/script.py
   - 确认引用无误后再执行命令，并捕获输出

用法说明：
- 命令在隔离的沙箱环境中运行
- 返回合并后的 stdout/stderr，并附带退出码
- 输出过大时可能被截断
- 重要：请避免使用 find / grep 等命令进行搜索；改用 grep / glob 工具。避免使用 cat/head/tail 读取文件；改用 read_file 工具。
- 多条命令请使用 ';' 或 '&&' 分隔；不要使用换行（引号内换行除外）
- 当命令间存在依赖时使用 '&&'（例如："mkdir dir && cd dir"）
- 当只需要顺序执行且不关心前序失败时使用 ';'
- 尽量使用绝对路径，避免频繁 cd 来保持工作目录稳定

示例：
推荐：
- execute(command="pytest /foo/bar/tests")
- execute(command="python /path/to/script.py")
- execute(command="npm install && npm test")

避免：
- execute(command="cd /foo/bar && pytest tests") # 请改用绝对路径
- execute(command="cat file.txt")                # 请改用 read_file
- execute(command="find . -name '*.py'")         # 请改用 glob
- execute(command="grep -r 'pattern' .")         # 请改用 grep
"""


# ========== 路径验证函数 ==========
def _validate_path(path: str, *, allowed_prefixes: list[str] | None = None) -> str:
    r"""验证并规范化文件路径以确保安全。

    通过防止目录遍历攻击和强制一致格式来确保路径安全可用。
    所有路径都会被规范化为使用正斜杠并以前导斜杠开头。

    此函数设计用于虚拟文件系统路径，会拒绝 Windows 绝对路径
    （如 C:/...、F:/...）以保持一致性并防止路径格式歧义。

    Args:
        path: 要验证和规范化的路径
        allowed_prefixes: 可选的允许路径前缀列表。如果提供，
            规范化后的路径必须以其中一个前缀开头

    Returns:
        规范化的标准路径，以 `/` 开头并使用正斜杠

    Raises:
        ValueError: 当路径包含遍历序列（`..` 或 `~`）、
            是 Windows 绝对路径（如 C:/...）、或不以允许的前缀开头时抛出
    """

    if ".." in path or path.startswith("~"):
        msg = f"Path traversal not allowed: {path}"
        raise ValueError(msg)

    # 拒绝 Windows 绝对路径（如 C:\...、D:/...）
    if re.match(r"^[a-zA-Z]:", path):
        msg = (
            f"Windows absolute paths are not supported: {path}. "
            "Please use virtual paths starting with / (e.g., /workspace/file.txt)"
        )
        raise ValueError(msg)

    normalized = os.path.normpath(path)
    normalized = normalized.replace("\\\\", "/")
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    if allowed_prefixes is not None and not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        msg = f"Path must start with one of {allowed_prefixes}: {path}"
        raise ValueError(msg)

    return normalized


def _get_backend(backend: RuntimeBackend | Callable[[], RuntimeBackend]) -> RuntimeBackend:
    """从后端实例或工厂函数获取解析后的后端实例。"""

    if callable(backend):
        return backend()
    return backend


class RuntimeBackendResolver:
    """运行时中间件（运行时解析器）。

    该类负责运行时的注册与解析：
    - 注册多个命名运行时后端（register_runtime）
    - 根据 runtime 参数解析 backend（resolve_backend / _resolve_backend）
    - 提供 runtime 参数描述文本（runtime_param_description / _runtime_param_description）

    注意：本类不负责构造工具集合；工具构造由本模块的 get_xxx_tool / get_client_tools 等函数承担。

    Args:
        default_runtime: 当未指定 runtime 参数时使用的默认运行时名称。
    """

    def __init__(self, default_runtime: str = "local") -> None:
        self._backends: dict[str, RuntimeBackend | Callable[[], RuntimeBackend]] = {}
        self._default_runtime = default_runtime

    @property
    def default_runtime(self) -> str:
        """默认运行时名称。"""

        return self._default_runtime

    def register_runtime(
        self,
        name: str,
        backend: RuntimeBackend | Callable[[], RuntimeBackend],
    ) -> "RuntimeBackendResolver":
        """注册一个命名运行时后端。

        Args:
            name: 运行时名称（如 "local"、"sandbox_1"）
            backend: 对应运行时的后端实例（或返回后端实例的工厂函数）

        Returns:
            self（便于链式调用）
        """

        if not name:
            raise ValueError("runtime name must be non-empty")
        self._backends[name] = backend
        return self

    def runtime_param_description(self) -> str:
        """返回 runtime 参数的描述文本（用于工具 schema）。"""

        return self._runtime_param_description()

    def resolve_backend(self, runtime: str) -> RuntimeBackend | str:
        """解析 runtime 对应的后端；未指定时回退到默认运行时。"""

        return self._resolve_backend(runtime)

    def _runtime_param_description(self) -> str:
        names = sorted(self._backends.keys())
        joined = ", ".join(names) if names else "(none)"
        return f"运行此工具的目标运行时（必填）。可选值: {joined}。"

    def _resolve_backend(self, runtime: str) -> RuntimeBackend | str:
        resolved_runtime = runtime

        if resolved_runtime not in self._backends:
            available = ", ".join(sorted(self._backends.keys()))
            return f"Error: Unknown runtime '{resolved_runtime}'. Available runtimes: {available or '(none)'}"

        return _get_backend(self._backends[resolved_runtime])


# ========== 工具生成器函数 ==========
def get_ls_tool(resolver: RuntimeBackendResolver, custom_description: str | None = None) -> BaseTool:
    """生成 ls（列出文件）工具。"""

    tool_description = custom_description or LIST_FILES_TOOL_DESCRIPTION

    def ls(
        path: Annotated[str, "Absolute path to the directory to list. Must be absolute, not relative."],
        target_runtime: str,
    ) -> str:
        """列出目录中的文件。"""

        resolved_backend = resolver.resolve_backend(target_runtime)
        if isinstance(resolved_backend, str):
            return resolved_backend

        validated_path = _validate_path(path)
        infos = resolved_backend.ls_info(validated_path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    ls.__annotations__["target_runtime"] = Annotated[str, resolver.runtime_param_description()]

    return StructuredTool.from_function(
        name="ls",
        description=tool_description,
        func=ls,
    )


def get_read_file_tool(resolver: RuntimeBackendResolver, custom_description: str | None = None) -> BaseTool:
    """生成 read_file 工具。"""

    tool_description = custom_description or READ_FILE_TOOL_DESCRIPTION

    def read_file(
        file_path: Annotated[str, "Absolute path to the file to read. Must be absolute, not relative."],
        target_runtime: str,
        offset: Annotated[int, "Line number to start reading from (0-indexed). Use for pagination of large files."] = (
            DEFAULT_READ_OFFSET
        ),
        limit: Annotated[int, "Maximum number of lines to read. Use for pagination of large files."] = (
            DEFAULT_READ_LIMIT
        ),
    ) -> str:
        """读取文件内容。"""

        resolved_backend = resolver.resolve_backend(target_runtime)
        if isinstance(resolved_backend, str):
            return resolved_backend

        validated_path = _validate_path(file_path)
        result = resolved_backend.read(validated_path, offset=offset, limit=limit)
        lines = result.splitlines(keepends=True)
        if len(lines) > limit:
            lines = lines[:limit]
        return "".join(lines)

    read_file.__annotations__["target_runtime"] = Annotated[str, resolver.runtime_param_description()]

    return StructuredTool.from_function(
        name="read_file",
        description=tool_description,
        func=read_file,
    )


def get_write_file_tool(resolver: RuntimeBackendResolver, custom_description: str | None = None) -> BaseTool:
    """生成 write_file 工具。"""

    tool_description = custom_description or WRITE_FILE_TOOL_DESCRIPTION

    def write_file(
        file_path: Annotated[str, "Absolute path where the file should be created. Must be absolute, not relative."],
        content: Annotated[str, "The text content to write to the file. This parameter is required."],
        target_runtime: str,
    ) -> str:
        """创建新文件。"""

        resolved_backend = resolver.resolve_backend(target_runtime)
        if isinstance(resolved_backend, str):
            return resolved_backend

        validated_path = _validate_path(file_path)
        res = resolved_backend.write(validated_path, content)
        if res.error:
            return res.error
        return f"Updated file {res.path}"

    write_file.__annotations__["target_runtime"] = Annotated[str, resolver.runtime_param_description()]

    return StructuredTool.from_function(
        name="write_file",
        description=tool_description,
        func=write_file,
    )


def get_edit_file_tool(resolver: RuntimeBackendResolver, custom_description: str | None = None) -> BaseTool:
    """生成 edit_file 工具。"""

    tool_description = custom_description or EDIT_FILE_TOOL_DESCRIPTION

    def edit_file(
        file_path: Annotated[str, "Absolute path to the file to edit. Must be absolute, not relative."],
        old_string: Annotated[
            str, "The exact text to find and replace. Must be unique in the file unless replace_all is True."
        ],
        new_string: Annotated[str, "The text to replace old_string with. Must be different from old_string."],
        target_runtime: str,
        replace_all: Annotated[
            bool, "If True, replace all occurrences of old_string. If False (default), old_string must be unique."
        ] = (False),
    ) -> str:
        """通过字符串替换编辑文件。"""

        resolved_backend = resolver.resolve_backend(target_runtime)
        if isinstance(resolved_backend, str):
            return resolved_backend

        validated_path = _validate_path(file_path)
        res = resolved_backend.edit(validated_path, old_string, new_string, replace_all=replace_all)
        if res.error:
            return res.error
        return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

    edit_file.__annotations__["target_runtime"] = Annotated[str, resolver.runtime_param_description()]

    return StructuredTool.from_function(
        name="edit_file",
        description=tool_description,
        func=edit_file,
    )


def get_glob_tool(resolver: RuntimeBackendResolver, custom_description: str | None = None) -> BaseTool:
    """生成 glob 工具。"""

    tool_description = custom_description or GLOB_TOOL_DESCRIPTION

    def glob(
        pattern: Annotated[str, "Glob pattern to match files (e.g., '**/*.py', '*.txt', '/subdir/**/*.md')."],
        target_runtime: str,
        path: Annotated[str, "Base directory to search from. Defaults to root '/'."] = "/",
    ) -> str:
        """查找匹配 glob 模式的文件。"""

        resolved_backend = resolver.resolve_backend(target_runtime)
        if isinstance(resolved_backend, str):
            return resolved_backend

        infos = resolved_backend.glob_info(pattern, path=path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    glob.__annotations__["target_runtime"] = Annotated[str, resolver.runtime_param_description()]

    return StructuredTool.from_function(
        name="glob",
        description=tool_description,
        func=glob,
    )


def get_grep_tool(resolver: RuntimeBackendResolver, custom_description: str | None = None) -> BaseTool:
    """生成 grep 工具。"""

    tool_description = custom_description or GREP_TOOL_DESCRIPTION

    def grep(
        pattern: Annotated[str, "Text pattern to search for (literal string, not regex)."],
        target_runtime: str,
        path: Annotated[str | None, "Directory to search in. Defaults to current working directory."] = None,
        glob: Annotated[str | None, "Glob pattern to filter which files to search (e.g., '*.py')."] = None,
        output_mode: Annotated[
            Literal["files_with_matches", "content", "count"],
            "Output format: 'files_with_matches' (file paths only, default), 'content' (matching lines with context), 'count' (match counts per file).",
        ] = "files_with_matches",
    ) -> str:
        """在文件中搜索文本模式。"""

        resolved_backend = resolver.resolve_backend(target_runtime)
        if isinstance(resolved_backend, str):
            return resolved_backend

        raw = resolved_backend.grep_raw(pattern, path=path, glob=glob)
        if isinstance(raw, str):
            return raw
        formatted = format_grep_matches(raw, output_mode)
        return truncate_if_too_long(formatted)

    grep.__annotations__["target_runtime"] = Annotated[str, resolver.runtime_param_description()]

    return StructuredTool.from_function(
        name="grep",
        description=tool_description,
        func=grep,
    )


def get_execute_tool(
    resolver: RuntimeBackendResolver,
    custom_description: str | None = None,
    enable_security: bool | None = None,
) -> BaseTool:
    """生成 execute 工具用于执行 shell 命令。

    Args:
        resolver: 运行时解析器。
        custom_description: 自定义工具描述。
        enable_security: 是否启用命令安全校验。
            默认为 None 时启用校验（True）。
            设为 False 可完全跳过安全校验（仅用于测试或迁移过渡期）。
    """

    from .security import validate_command

    # 确定是否启用安全校验（默认启用）
    if enable_security is None:
        enable_security = True

    tool_description = custom_description or EXECUTE_TOOL_DESCRIPTION

    def execute(
        command: Annotated[str, "Shell command to execute in the sandbox environment."],
        target_runtime: str,
    ) -> str:
        """在沙箱环境中执行 shell 命令。"""

        resolved_backend = resolver.resolve_backend(target_runtime)
        if isinstance(resolved_backend, str):
            return resolved_backend

        # 【新增】安全校验：命令白名单检查
        if enable_security:
            result = validate_command(command)
            if not result.is_allowed:
                return f"命令执行被拒绝：{result.reason}"

        result = resolved_backend.execute(command)

        parts = [result.output]
        if result.truncated:
            parts.append("\n[Output was truncated due to size limits]")
        return "".join(parts)

    async def async_execute(
        command: Annotated[str, "Shell command to execute in the sandbox environment."],
        target_runtime: str,
    ) -> str:
        """异步执行 shell 命令。"""

        resolved_backend = resolver.resolve_backend(target_runtime)
        if isinstance(resolved_backend, str):
            return resolved_backend

        # 【新增】安全校验：命令白名单检查
        if enable_security:
            result = validate_command(command)
            if not result.is_allowed:
                return f"命令执行被拒绝：{result.reason}"

        result = await resolved_backend.aexecute(command)

        parts = [result.output]
        if result.truncated:
            parts.append("\n[Output was truncated due to size limits]")
        return "".join(parts)

    execute.__annotations__["target_runtime"] = Annotated[str, resolver.runtime_param_description()]
    async_execute.__annotations__["target_runtime"] = Annotated[str, resolver.runtime_param_description()]

    return StructuredTool.from_function(
        name="execute",
        description=tool_description,
        func=execute,
        coroutine=async_execute,
    )


# ========== 客户端工具集合 ==========


def get_client_tools_with_runtime(
    resolver: RuntimeBackendResolver,
    custom_tool_descriptions: dict[str, str] | None = None,
    enable_security: bool | None = None,
) -> list[BaseTool]:
    """构造客户端工具集合。

    返回固定 7 个工具：ls、read_file、write_file、edit_file、glob、grep、execute。

    Args:
        resolver: 运行时解析器（负责 runtime -> backend 路由）。
        custom_tool_descriptions: 可选的自定义工具描述字典，key 为工具名。
        enable_security: 是否启用 execute 工具的命令安全校验。
            默认为 None，从环境变量读取。设为 False 可跳过校验。

    Returns:
        LangChain 工具列表。
    """

    if custom_tool_descriptions is None:
        custom_tool_descriptions = {}

    return [
        get_ls_tool(resolver, custom_tool_descriptions.get("ls")),
        get_read_file_tool(resolver, custom_tool_descriptions.get("read_file")),
        get_write_file_tool(resolver, custom_tool_descriptions.get("write_file")),
        get_edit_file_tool(resolver, custom_tool_descriptions.get("edit_file")),
        get_glob_tool(resolver, custom_tool_descriptions.get("glob")),
        get_grep_tool(resolver, custom_tool_descriptions.get("grep")),
        get_execute_tool(resolver, custom_tool_descriptions.get("execute"), enable_security=enable_security),
    ]
