# -*- coding: utf-8 -*-
"""aidev_agent.core.tools.runtime_tools.security

TencentBlueKing is pleased to support the open source community by making
蓝鲸智云 - AIDev (BlueKing - AIDev) available.
Copyright (C) 2025 THL A29 Limited,
a Tencent company. All rights reserved.
Licensed under the MIT License (the " License ");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the License for the
specific language governing permissions and limitations under the License.
We undertake not to change the open source license (MIT license) applicable
to the current version of the project delivered to anyone in the future.

命令白名单安全校验模块。

本模块提供 bash 命令的白名单校验机制，确保只有通过校验的命令才能被执行。
使用 bashlex 作为 bash 语法解析器，支持递归解析 bash -c 参数、
按 shell 操作符拆分命令、检测禁止的操作符等。

公开接口：
- validate_command(): 校验命令是否允许执行，返回详细结果
- is_command_allowed(): 快捷方法，返回命令是否允许
- ValidationResult: 校验结果数据类
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import bashlex
import bashlex.ast as bashast

logger = logging.getLogger(__name__)


# ========== 校验结果 ==========


@dataclass
class ValidationResult:
    """命令校验结果。"""

    is_allowed: bool
    """命令是否允许执行"""

    reason: str = ""
    """拒绝原因（当 is_allowed 为 False 时有效）"""

    rejected_command: str = ""
    """被拒绝的具体命令（用于审计日志）"""

    def __bool__(self) -> bool:
        return self.is_allowed


# ========== 白名单命令定义 ==========

# 类别一：系统信息查询
_SYSTEM_INFO_COMMANDS: frozenset[str] = frozenset(
    {
        "pwd",
        "date",
        "hostname",
        "whoami",
        "id",
        "uptime",
        "uname",
        "free",
        "df",
    }
)

# 类别二：文件/目录操作
_FILE_DIR_COMMANDS: frozenset[str] = frozenset(
    {
        "ls",
        "dir",
        "cd",
        "stat",
        "readlink",
        "file",
    }
)

# 类别三：文件内容操作
_FILE_CONTENT_COMMANDS: frozenset[str] = frozenset(
    {
        "cat",
        "head",
        "tail",
        "grep",
        "egrep",
        "wc",
        "sort",
        "uniq",
        "cut",
        "tr",
        "awk",
        "sed",
        "diff",
    }
)

# 类别四：基础工具
_BASIC_TOOL_COMMANDS: frozenset[str] = frozenset(
    {
        "echo",
        "printf",
        "true",
        "false",
        "sleep",
        "clear",
        "reset",
    }
)

# 类别五：压缩/解压
_ARCHIVE_COMMANDS: frozenset[str] = frozenset(
    {
        "tar",
        "gzip",
        "zip",
    }
)

# 类别六：脚本执行
_SCRIPT_COMMANDS: frozenset[str] = frozenset(
    {
        "bash",
        "sh",
        "zsh",
        "python",
        "python3",
    }
)

# 完整白名单命令集合
ALLOWED_COMMANDS: frozenset[str] = (
    _SYSTEM_INFO_COMMANDS
    | _FILE_DIR_COMMANDS
    | _FILE_CONTENT_COMMANDS
    | _BASIC_TOOL_COMMANDS
    | _ARCHIVE_COMMANDS
    | _SCRIPT_COMMANDS
)

# ========== 命令参数限制 ==========

# uname 仅允许的参数集合（"" 表示允许无参数）
_UNAME_ALLOWED_FLAGS: frozenset[str] = frozenset({"", "-s", "-v"})

# df 仅允许的参数集合
_DF_ALLOWED_FLAGS: frozenset[str] = frozenset({"", "-h"})


class ParameterRestriction:
    """参数限制基类。"""

    def is_allowed(self, args: list[str]) -> tuple[bool, str]:
        """检查参数是否符合限制。

        Args:
            args: 命令参数列表。

        Returns:
            (是否允许, 拒绝原因) 元组。
        """
        raise NotImplementedError


@dataclass
class AllowedFlagsOnly(ParameterRestriction):
    """仅允许指定的标志参数。其他标志参数拒绝，非标志参数（不以 - 开头）允许。"""

    flags: frozenset[str]
    """允许的 flag 集合，"" 表示允许无参数（即纯非 flag 参数场景）"""

    def is_allowed(self, args: list[str]) -> tuple[bool, str]:
        for arg in args:
            # 非标志参数（不以 - 开头）一律放行
            if not arg.startswith("-"):
                continue
            # 长选项如 --help 不在允许列表中则拒绝
            if arg not in self.flags:
                return False, f"不允许使用参数 '{arg}'"
        return True, ""


@dataclass
class ForbiddenFlags(ParameterRestriction):
    """禁止指定的标志，其他允许。"""

    forbidden: frozenset[str]

    def is_allowed(self, args: list[str]) -> tuple[bool, str]:
        for arg in args:
            if arg in self.forbidden:
                return False, f"不允许使用参数 '{arg}'"
        return True, ""


# 命令参数限制映射
PARAMETER_RESTRICTIONS: dict[str, ParameterRestriction] = {
    "uname": AllowedFlagsOnly(flags=_UNAME_ALLOWED_FLAGS),
    "df": AllowedFlagsOnly(flags=_DF_ALLOWED_FLAGS),
}

# ========== 禁止的操作符/模式 ==========

# 需要在命令解析阶段检测的禁止操作符
# 注意：| 作为管道符会在 bashlex 解析时拆分，此处用于原始文本预检测
_REJECTED_OPERATORS_PATTERN = re.compile(
    r"""
    (?:
        # 后台执行符（不是管道符的一部分）
        (?<!\|)\s*&\s*(?!\&)   # 单独的 &（非 &&）
        |
        #  Here Document / Here String
        <<<|<<(?!\()            # <<< 或 <<（但不是 <( 进程替换）
        |
        # 进程替换
        <\(|>\(
        |
        # 命令替换：$(cmd) 或 `cmd`
        \$\(|`(?!\')
    )
    """,
    re.VERBOSE,
)

# 需要在拆分子命令后，对每个子命令检测的禁止操作符（原始文本层面）
# 这些操作符如果出现在命令中（非引号内），应被拒绝
_REJECTED_OPERATORS_IN_COMMAND = re.compile(
    r"""
    (?:
        # 后台执行相关
        (?<!\|)\s*&\s*(?!\&)   # 单独的 &（非 &&）
        |
        # nohup（无论是否配合 &）
        \bnohup\b
        |
        # setsid / disown / screen / tmux
        \bsetsid\b
        |
        \bdisown\b
        |
        \bscreen\b
        |
        \btmux\b
        |
        # Here Document / Here String
        <<<
        |
        # 进程替换
        <\(|>\(
    )
    """,
    re.VERBOSE,
)

# 重定向操作符检测（在拆分子命令后检测）
_REDIRECT_PATTERN = re.compile(
    r"""
    (?:
        \d?>|>>|>\&    # 输出重定向：> >> 2> &>
        |
        <(?!\()         # 输入重定向 <（但不是 <( 进程替换）
        |
        <<<|<<          # Here doc/string
    )
    """,
    re.VERBOSE,
)

# 命令替换检测
_COMMAND_SUBSTITUTION_PATTERN = re.compile(r"\$\(|`[^`]*`")

# 重定向到 /dev/null 的安全模式（允许放行）
# 匹配形式：>/dev/null、2>/dev/null、&>/dev/null、>>/dev/null、2>>/dev/null 等
_REDIRECT_TO_DEV_NULL_PATTERN = re.compile(
    r"""
    (?:
        (?:\d|&)?>>?\s*/dev/null   # [n]>/dev/null 或 [n]>>/dev/null 或 &>/dev/null
    )
    """,
    re.VERBOSE,
)

# 大括号扩展检测
_BRACE_EXPANSION_PATTERN = re.compile(r"\{[^{}]*,[^{}]*\}")

# ========== 脚本文件扩展名 ==========

_SCRIPT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".sh",
        ".py",
        ".pl",
        ".rb",
        ".js",
    }
)

# ========== 配置 ==========


def _get_env_list(name: str, default: list[str] | None = None) -> list[str]:
    """从环境变量读取逗号分隔的列表。"""
    value = os.environ.get(name)
    if value is None:
        return default if default is not None else []
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_env_bool(name: str, default: bool = True) -> bool:
    """从环境变量读取布尔值。"""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


# 默认允许的脚本目录
DEFAULT_ALLOWED_SCRIPT_DIRS: list[str] = _get_env_list(
    "AIDEV_ALLOWED_SCRIPT_DIRS",
    default=["/workspace", "/home", "/tmp", "/app"],
)

# 额外放行的命令（谨慎使用）
_ADDITIONAL_ALLOWED_COMMANDS: frozenset[str] = frozenset(_get_env_list("AIDEV_ADDITIONAL_ALLOWED_COMMANDS", default=[]))

# 合并后的白名单（基础白名单 + 额外命令）
EFFECTIVE_ALLOWED_COMMANDS: frozenset[str] = ALLOWED_COMMANDS | _ADDITIONAL_ALLOWED_COMMANDS


# ========== 命令名规范化 ==========


def _normalize_command_name(raw_name: str) -> str:
    """规范化命令名：提取 basename、拒绝路径遍历、处理特殊形式。

    Args:
        raw_name: 原始命令名（可能包含路径）。

    Returns:
        规范化后的纯命令名。

    Raises:
        ValueError: 当命令名包含路径遍历或其他不安全模式时。
    """
    name = raw_name.strip()

    if not name:
        raise ValueError("空命令名")

    # 拒绝包含路径遍历的命令
    if ".." in name:
        raise ValueError(f"命令名包含路径遍历: {raw_name}")

    # 如果是绝对路径，提取 basename
    if name.startswith("/"):
        normalized = os.path.normpath(name)
        if ".." in normalized:
            raise ValueError(f"命令路径包含遍历: {raw_name}")
        return os.path.basename(normalized)

    # 如果是相对路径（以 ./ 或 ../ 开头）
    if name.startswith(("./", "../")) or ("/" in name and not name.startswith("-")):
        normalized = os.path.normpath(name)
        if ".." in normalized:
            raise ValueError(f"命令路径包含遍历: {raw_name}")
        return os.path.basename(normalized)

    # 纯命令名，直接返回
    return name


# ========== bashlex 解析辅助 ==========


def _extract_command_parts(node: Any) -> tuple[str, list[str]]:
    """从 bashlex CommandNode 中提取命令名和参数列表。

    正确处理引号、转义等 shell 语法，提取字面值。

    Args:
        node: bashlex CommandNode。

    Returns:
        (命令名, 参数列表) 元组。
    """
    parts: list[str] = []
    cmd_name: str = ""

    for child in node.parts:
        word = _node_to_string(child) if isinstance(child, bashast.node) else str(child)

        if not cmd_name:
            cmd_name = word
        else:
            parts.append(word)

    return cmd_name, parts


def _node_to_string(node: Any) -> str:
    """将 bashlex AST 节点转换为字符串字面值。

    处理 WordNode、ParameterNode 等节点类型，正确展开
    引号内的内容和变量引用。

    Args:
        node: bashlex AST 节点。

    Returns:
        节点的字符串字面值。
    """
    kind = getattr(node, "kind", None)

    if kind == "word":
        # WordNode：可能包含引号、转义等
        # 需要提取其中的 word 属性或 parts
        if hasattr(node, "word"):
            return node.word
        if hasattr(node, "parts") and node.parts:
            return "".join(_part_to_string(p) for p in node.parts)
        return str(node)

    elif kind == "parameter":
        # 变量引用如 $HOME、${VAR}
        if hasattr(node, "name"):
            name = node.name
            return f"${{{name}}}"
        return str(node)

    elif kind == "commandsubstitution":
        # 命令替换 $(cmd) 或 `cmd`
        return "$(cmd)"

    elif hasattr(node, "word"):
        return node.word

    elif hasattr(node, "parts") and node.parts:
        return "".join(_part_to_string(p) for p in node.parts)

    return str(node)


def _part_to_string(part: Any) -> str:
    """将 bashlex 节点的一部分转换为字符串。"""
    if isinstance(part, str):
        return part
    if hasattr(part, "word"):
        return part.word
    if hasattr(part, "parts") and part.parts:
        return "".join(_part_to_string(p) for p in part.parts)
    return str(part)


# ========== 命令拆分 ==========


def _split_by_shell_operators(command: str) -> list[str]:
    """使用 bashlex 将命令按 shell 操作符拆分为子命令列表。

    正确处理引号内的操作符（不作为分隔符）。
    使用 AST 节点的 pos 属性从原始命令字符串中切片，保留引号等原始语法。

    Args:
        command: 待拆分的命令字符串。

    Returns:
        子命令字符串列表。空子命令会被过滤。
    """
    sub_commands: list[str] = []

    try:
        nodes = bashlex.parse(command)
    except bashlex.errors.ParsingError as e:
        # 解析失败，可能是语法错误或引号不匹配
        raise ValueError(f"命令解析失败: {e}")

    for node in nodes:
        kind = getattr(node, "kind", None)

        if kind == "list":
            # list 节点包含多个通过操作符连接的命令
            current_parts: list[Any] = []
            for part in node.parts:
                part_kind = getattr(part, "kind", None)
                if part_kind == "operator":
                    # 遇到操作符，保存当前积累的命令
                    cmd_str = _nodes_to_command_string(current_parts, command)
                    if cmd_str.strip():
                        sub_commands.append(cmd_str.strip())
                    current_parts = []
                elif part_kind == "command" or part_kind == "pipe":
                    current_parts.append(part)
                else:
                    current_parts.append(part)
            # 保存最后积累的命令
            if current_parts:
                cmd_str = _nodes_to_command_string(current_parts, command)
                if cmd_str.strip():
                    sub_commands.append(cmd_str.strip())

        elif kind == "command":
            # 单个命令
            cmd_str = _node_to_original_string(node, command)
            if cmd_str.strip():
                sub_commands.append(cmd_str.strip())

        elif kind == "pipeline":
            # 管道命令：拆分为多个子命令
            for part in node.parts:
                part_kind = getattr(part, "kind", None)
                if part_kind == "command":
                    cmd_str = _node_to_original_string(part, command)
                    if cmd_str.strip():
                        sub_commands.append(cmd_str.strip())
                elif part_kind == "pipe":
                    continue  # 跳过管道符本身
                else:
                    cmd_str = _node_to_original_string(part, command)
                    if cmd_str.strip():
                        sub_commands.append(cmd_str.strip())

        else:
            # 其他类型节点，尝试转换为字符串
            cmd_str = _node_to_original_string(node, command)
            if cmd_str.strip():
                sub_commands.append(cmd_str.strip())

    return [c for c in sub_commands if c.strip()]


def _nodes_to_command_string(nodes: list[Any], original_command: str) -> str:
    """将多个 AST 节点组合为命令字符串，使用 pos 从原始命令切片。"""
    if not nodes:
        return ""
    # 取所有节点的最小 start 和最大 end，从原始命令中切片
    starts: list[int] = []
    ends: list[int] = []
    for node in nodes:
        if hasattr(node, "pos"):
            starts.append(node.pos[0])
            ends.append(node.pos[1])
    if starts and ends:
        return original_command[min(starts) : max(ends)]
    # fallback：逐个节点切片再拼接
    parts_strs: list[str] = []
    for node in nodes:
        parts_strs.append(_node_to_original_string(node, original_command))
    return " ".join(parts_strs)


def _node_to_original_string(node: Any, original_command: str) -> str:
    """从 AST 节点还原原始命令字符串。

    使用节点的 pos 属性从原始命令字符串中切片，保留引号等原始语法。
    """
    if hasattr(node, "pos"):
        start, end = node.pos
        return original_command[start:end]

    # fallback for nodes without pos
    kind = getattr(node, "kind", None)

    if kind == "pipe":
        return "|"

    if hasattr(node, "word"):
        return node.word

    if hasattr(node, "parts") and node.parts:
        return "".join(_part_to_string(p) for p in node.parts)

    return str(node)


# ========== bash -c 检测与提取 ==========


def _is_bash_c_form(cmd_name: str, args: list[str]) -> bool:
    """判断是否为 bash -c "command" 形式。

    Args:
        cmd_name: 规范化后的命令名。
        args: 参数列表。

    Returns:
        是否为 bash -c 形式。
    """
    if cmd_name not in ("bash", "sh", "zsh"):
        return False
    # 检查是否有 -c 参数
    return "-c" in args


def _extract_bash_c_content(cmd_name: str, args: list[str]) -> str | None:
    """从 bash -c 形式的命令中提取要执行的内部命令。

    Args:
        cmd_name: 规范化后的命令名。
        args: 参数列表。

    Returns:
        内部命令字符串，如果未找到则返回 None。
    """
    if not _is_bash_c_form(cmd_name, args):
        return None

    try:
        c_index = args.index("-c")
        if c_index + 1 < len(args):
            return args[c_index + 1]
    except ValueError:
        pass
    return None


# ========== 禁止模式检测 ==========


def _check_rejected_patterns(command: str) -> tuple[bool, str]:
    """检测命令中是否包含禁止的操作符或模式。

    在 bashlex 解析之前进行预检测，捕获需要在原始文本层面
    拒绝的模式（如命令替换、进程替换、后台执行、重定向等）。

    Args:
        command: 原始命令字符串。

    Returns:
        (是否包含禁止模式, 拒绝原因) 元组。
    """
    # 检测命令替换 $(cmd) 或 `cmd`
    if _COMMAND_SUBSTITUTION_PATTERN.search(command):
        return False, "命令中包含命令替换语法（$(...) 或 `...`），不允许执行"

    # 检测进程替换 <(...) 或 >(...)
    if re.search(r"<\(|>\(", command):
        return False, "命令中包含进程替换语法（<() 或 >()），不允许执行"

    # 检测 Here String <<<
    if re.search(r"<<<", command):
        return False, "命令中包含 Here String 语法（<<<），不允许执行"

    # 检测 Here Document <<
    if re.search(r"<<(?!<<)(?!\()", command):
        return False, "命令中包含 Here Document 语法（<<），不允许执行"

    # 检测重定向操作符（>、>>、< 等）
    # 先排除所有重定向到 /dev/null 的安全模式，再检测是否还有其他重定向
    if _REDIRECT_PATTERN.search(command):
        # 将重定向到 /dev/null 的部分临时移除后，再检测是否还存在其他重定向
        sanitized = _REDIRECT_TO_DEV_NULL_PATTERN.sub("", command)
        if _REDIRECT_PATTERN.search(sanitized):
            return False, "命令中包含重定向操作符（>、>>、< 等），不允许使用重定向"

    # 检测 |& 操作符（管道+stderr 重定向）
    if re.search(r"\|\&", command):
        return False, "命令中包含 |& 操作符，不允许执行"

    # 检测后台执行符 &
    # 需要排除 &&（逻辑与）和 |& 的情况
    # 匹配单独的 &，不在 && 或 |& 中
    if re.search(r"(?<![&\|])\&(?!\&)", command):
        return False, "命令中包含后台执行符（&），不允许在后台执行命令"

    # 检测 nohup
    if re.search(r"\bnohup\b", command):
        return False, "命令中包含 nohup，不允许脱离终端执行"

    # 检测 setsid
    if re.search(r"\bsetsid\b", command):
        return False, "命令中包含 setsid，不允许新建会话执行"

    # 检测 disown
    if re.search(r"\bdisown\b", command):
        return False, "命令中包含 disown，不允许脱离终端"

    # 检测 screen / tmux
    if re.search(r"\bscreen\b", command):
        return False, "命令中包含 screen，不允许使用终端复用器"

    if re.search(r"\btmux\b", command):
        return False, "命令中包含 tmux，不允许使用终端复用器"

    # 检测大括号扩展
    if _BRACE_EXPANSION_PATTERN.search(command):
        return False, "命令中包含大括号扩展语法（{}），不允许执行"

    return True, ""


def _check_redirection_in_command(command: str) -> tuple[bool, str]:
    """检测命令中是否包含重定向操作符。

    在拆分子命令后对每个子命令检测。

    Args:
        command: 子命令字符串。

    Returns:
        (是否包含重定向, 拒绝原因) 元组。
    """
    if _REDIRECT_PATTERN.search(command):
        # 放行重定向到 /dev/null 的安全模式
        sanitized = _REDIRECT_TO_DEV_NULL_PATTERN.sub("", command)
        if _REDIRECT_PATTERN.search(sanitized):
            return False, "命令中包含重定向操作符（>、>>、< 等），不允许使用重定向"
    return True, ""


# ========== 脚本路径校验 ==========


def _is_script_file(path: str) -> bool:
    """判断路径是否指向脚本文件。

    Args:
        path: 文件路径。

    Returns:
        是否为脚本文件。
    """
    ext = os.path.splitext(path)[1].lower()
    return ext in _SCRIPT_EXTENSIONS


def _check_script_path_allowed(
    script_path: str,
    allowed_dirs: list[str],
) -> tuple[bool, str]:
    """检查脚本路径是否在允许的目录内。

    Args:
        script_path: 脚本文件路径。
        allowed_dirs: 允许的目录列表。

    Returns:
        (是否允许, 拒绝原因) 元组。
    """
    if not script_path:
        return False, "未指定脚本路径"

    # 规范化路径
    normalized = os.path.normpath(script_path)

    # 拒绝绝对路径中的遍历
    if ".." in normalized:
        return False, f"脚本路径包含目录遍历: {script_path}"

    # 检查是否在允许的目录内
    for allowed_dir in allowed_dirs:
        allowed_normalized = os.path.normpath(allowed_dir)
        # 路径必须是 allowed_dir 的直接或间接子路径
        if normalized == allowed_normalized:
            return True, ""
        if normalized.startswith(allowed_normalized + os.sep):
            return True, ""

    return False, f"脚本路径不在允许的目录内（允许: {', '.join(allowed_dirs)}）"


# ========== 单条命令校验 ==========


def _validate_single_command(
    sub_cmd: str,
    allowed_dirs: list[str],
    recursion_depth: int = 0,
    visited: set[str] | None = None,
) -> ValidationResult:
    """校验单条命令（可能是 bash -c 形式或普通命令）。

    这是递归校验的核心函数。

    Args:
        sub_cmd: 子命令字符串。
        allowed_dirs: 允许的脚本目录列表。
        recursion_depth: 递归深度（防止无限递归）。
        visited: 已访问的命令集合（防止循环）。

    Returns:
        校验结果。
    """
    if visited is None:
        visited = set()

    # 防止无限递归
    if recursion_depth > 10:
        return ValidationResult(
            is_allowed=False,
            reason="命令嵌套层级过深（超过 10 层），可能存在递归攻击",
            rejected_command=sub_cmd,
        )

    # 防止循环
    cmd_key = sub_cmd.strip()
    if cmd_key in visited:
        return ValidationResult(
            is_allowed=False,
            reason=f"检测到循环命令引用: {sub_cmd}",
            rejected_command=sub_cmd,
        )
    visited = visited | {cmd_key}

    # Step 1: 去除首尾空白
    sub_cmd = sub_cmd.strip()
    if not sub_cmd:
        return ValidationResult(
            is_allowed=False,
            reason="空命令",
            rejected_command=sub_cmd,
        )

    # Step 2: 检测注释（# 开头的命令或纯注释）
    if sub_cmd.startswith("#"):
        return ValidationResult(
            is_allowed=False,
            reason="注释不是有效命令",
            rejected_command=sub_cmd,
        )

    # Step 3: 检测禁止模式（命令替换、进程替换、后台执行、重定向等）
    ok, reason = _check_rejected_patterns(sub_cmd)
    if not ok:
        return ValidationResult(
            is_allowed=False,
            reason=reason,
            rejected_command=sub_cmd,
        )

    # Step 4: 使用 bashlex 解析命令
    try:
        nodes = bashlex.parse(sub_cmd)
    except bashlex.errors.ParsingError as e:
        return ValidationResult(
            is_allowed=False,
            reason=f"命令语法错误（可能是引号不匹配）: {e}",
            rejected_command=sub_cmd,
        )

    if not nodes:
        return ValidationResult(
            is_allowed=False,
            reason="无法解析命令",
            rejected_command=sub_cmd,
        )

    # Step 5: 处理多节点情况或包含操作符/pipeline 的命令
    # 如果有多个顶层节点、存在操作符、或是 pipeline，需要递归拆分
    needs_splitting = False
    for node in nodes:
        kind = getattr(node, "kind", None)
        if kind == "pipeline":
            needs_splitting = True
            break
        if kind == "list":
            for part in node.parts:
                if getattr(part, "kind", None) == "operator":
                    needs_splitting = True
                    break
        if needs_splitting:
            break

    if len(nodes) > 1 or needs_splitting:
        # 需要按操作符拆分
        try:
            sub_commands = _split_by_shell_operators(sub_cmd)
        except ValueError as e:
            return ValidationResult(
                is_allowed=False,
                reason=str(e),
                rejected_command=sub_cmd,
            )

        # 如果拆分后只有一条命令且和原命令相同，继续校验该命令
        if len(sub_commands) == 1 and sub_commands[0] == sub_cmd:
            pass  # 继续下面的单命令校验
        else:
            # 递归校验每个子命令
            for sc in sub_commands:
                sc = sc.strip()
                if not sc:
                    continue
                # 先检测注释
                if sc.startswith("#"):
                    continue
                result = _validate_single_command(
                    sc,
                    allowed_dirs,
                    recursion_depth + 1,
                    visited,
                )
                if not result.is_allowed:
                    return result
            return ValidationResult(is_allowed=True)

    # Step 6: 提取命令名和参数
    # 找到第一个 command 节点
    cmd_node = None
    for node in nodes:
        if getattr(node, "kind", None) == "command":
            cmd_node = node
            break

    if cmd_node is None:
        # 尝试从 list 节点中提取
        for node in nodes:
            if getattr(node, "kind", None) == "list":
                for part in node.parts:
                    if getattr(part, "kind", None) == "command":
                        cmd_node = part
                        break
                if cmd_node:
                    break

    if cmd_node is None:
        return ValidationResult(
            is_allowed=False,
            reason="无法识别命令结构",
            rejected_command=sub_cmd,
        )

    try:
        raw_cmd_name, args = _extract_command_parts(cmd_node)
    except Exception as e:
        return ValidationResult(
            is_allowed=False,
            reason=f"命令解析失败: {e}",
            rejected_command=sub_cmd,
        )

    if not raw_cmd_name:
        return ValidationResult(
            is_allowed=False,
            reason="无法提取命令名",
            rejected_command=sub_cmd,
        )

    # Step 7: 规范化命令名
    try:
        cmd_name = _normalize_command_name(raw_cmd_name)
    except ValueError as e:
        return ValidationResult(
            is_allowed=False,
            reason=str(e),
            rejected_command=sub_cmd,
        )

    # Step 8: 查白名单
    if cmd_name not in EFFECTIVE_ALLOWED_COMMANDS:
        return ValidationResult(
            is_allowed=False,
            reason=f"命令 '{cmd_name}' 不在允许列表中，如确需使用请联系管理员添加",
            rejected_command=sub_cmd,
        )

    # Step 9: 检查参数限制
    if cmd_name in PARAMETER_RESTRICTIONS:
        restriction = PARAMETER_RESTRICTIONS[cmd_name]
        ok, reason = restriction.is_allowed(args)
        if not ok:
            return ValidationResult(
                is_allowed=False,
                reason=f"命令 '{cmd_name}' {reason}",
                rejected_command=sub_cmd,
            )

    # Step 10: 特殊命令额外检查

    # Python / Python3：放行所有形式（脚本文件、-m module、-c 内联代码等）
    # （暂时 不再限制 -c 参数， 不要移除注释）
    # if cmd_name in ("python", "python3"):
    #     if "-c" in args:
    #         return ValidationResult(
    #             is_allowed=False,
    #             reason=f"命令 '{cmd_name} -c' 不允许执行内联代码",
    #             rejected_command=sub_cmd,
    #         )

    # Bash / Sh / Zsh：处理 -c 递归和脚本路径
    elif cmd_name in ("bash", "sh", "zsh"):
        bash_c_content = _extract_bash_c_content(cmd_name, args)
        if bash_c_content is not None:
            # 递归校验 bash -c 内部命令
            logger.debug("递归校验 bash -c 内容: %s", bash_c_content)
            result = _validate_single_command(
                bash_c_content,
                allowed_dirs,
                recursion_depth + 1,
                visited,
            )
            if not result.is_allowed:
                # 内部命令已被拒绝，直接返回其拒绝原因（避免重复前缀）
                return ValidationResult(
                    is_allowed=False,
                    reason=result.reason,
                    rejected_command=f"{cmd_name} -c '{bash_c_content}'",
                )
        elif "-c" in args:
            # bash -c 但缺少要执行的命令内容
            return ValidationResult(
                is_allowed=False,
                reason=f"命令 '{cmd_name} -c' 缺少要执行的命令内容",
                rejected_command=sub_cmd,
            )
        else:
            # 直接执行脚本文件
            script_path = _extract_script_argument(args)
            if script_path:
                ok, reason = _check_script_path_allowed(script_path, allowed_dirs)
                if not ok:
                    return ValidationResult(
                        is_allowed=False,
                        reason=f"Shell 脚本{reason}",
                        rejected_command=sub_cmd,
                    )

    return ValidationResult(is_allowed=True)


def _extract_script_argument(args: list[str]) -> str | None:
    """从参数列表中提取脚本文件路径。

    跳过已知的标志参数，取第一个非标志参数作为脚本路径。

    Args:
        args: 参数列表。

    Returns:
        脚本路径，如果未找到则返回 None。
    """
    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        # 跳过 -c 及其参数
        if arg == "-c":
            skip_next = True
            continue
        # 跳过其他可能带值的标志
        if arg in ("-o", "--options", "-O"):
            skip_next = True
            continue
        # 非标志参数，认为是脚本路径
        if not arg.startswith("-"):
            return arg
    return None


# ========== 输出脱敏 ==========

_REDACTED_PLACEHOLDER = "__BKAI_AGENT_REDACTED__"


def redact_output(text: str, sensitive_values: list[str]) -> str:
    """对输出文本进行脱敏处理，将敏感值替换为占位符。

    对输入文本执行精确字符串匹配，将所有出现的敏感值替换为
    `__BKAI_AGENT_REDACTED__`。仅执行精确匹配，不使用正则表达式。

    Args:
        text: 待脱敏的输出文本
        sensitive_values: 需要脱敏的敏感值列表，列表中的每个值
            都会在文本中被精确匹配并替换

    Returns:
        脱敏后的文本，敏感值已被替换为 `__BKAI_AGENT_REDACTED__`

    Example:
        >>> redact_output("token is abc123", ["abc123"])
        'token is __BKAI_AGENT_REDACTED__'
        >>> redact_output("no secrets here", ["secret_token"])
        'no secrets here'
    """
    if not sensitive_values:
        return text

    result = text
    for value in sensitive_values:
        if value and value in result:
            result = result.replace(value, _REDACTED_PLACEHOLDER)
    return result


# ========== 路径验证 ==========


def validate_path(path: str, *, allowed_prefixes: list[str] | None = None) -> str:
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
        规范化的标准路径，避免 a/../../b 这种情况出现

    Raises:
        ValueError: 当路径包含遍历序列（`..`）、
            是 Windows 绝对路径（如 C:/...）、或不以允许的前缀开头时抛出
    """

    # 拒绝 Windows 绝对路径（如 C:\...、D:/...）
    if re.match(r"^[a-zA-Z]:", path):
        msg = (
            f"Windows absolute paths are not supported: {path}. "
            "Please use virtual paths starting with / (e.g., /workspace/file.txt)"
        )
        raise ValueError(msg)

    normalized = os.path.normpath(path)
    normalized = normalized.replace("\\", "/")

    # 先规范化再检查路径遍历，避免 a/../b 被误拒（normpath 后为安全的 b）
    if ".." in normalized.split("/"):
        msg = f"Path traversal not allowed: {path}"
        raise ValueError(msg)

    if allowed_prefixes is not None and not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        msg = f"Path must start with one of {allowed_prefixes}: {path}"
        raise ValueError(msg)

    return normalized


# ========== 公开接口 ==========


def validate_command(
    command: str,
    *,
    allowed_script_dirs: list[str] | None = None,
    strict: bool = True,
) -> ValidationResult:
    """校验命令是否允许执行。

    对输入的命令字符串进行完整的白名单校验，包括：
    - 禁止模式检测（命令替换、进程替换、后台执行等）
    - 按 shell 操作符拆分并递归校验子命令
    - 命令名白名单检查
    - 参数限制检查
    - 特殊命令（python、bash 等）额外检查
    - bash -c 内部命令递归校验

    Args:
        command: 待校验的命令字符串。
        allowed_script_dirs: 允许执行脚本的目录列表。
            默认为 DEFAULT_ALLOWED_SCRIPT_DIRS（可从环境变量配置）。
        strict: 严格模式。当前保留参数，未来可能用于放宽限制。

    Returns:
        ValidationResult: 包含 is_allowed 和 reason 的校验结果。

    Example:
        >>> result = validate_command("ls /tmp")
        >>> result.is_allowed
        True
        >>> result = validate_command("rm file.txt")
        >>> result.is_allowed
        False
        >>> result.reason
        "命令 'rm' 不在允许列表中..."
    """
    if allowed_script_dirs is None:
        allowed_script_dirs = DEFAULT_ALLOWED_SCRIPT_DIRS

    logger.debug("开始校验命令: %s", command)

    # 预处理：去除首尾空白，检测空字节
    command = command.strip()
    if not command:
        return ValidationResult(
            is_allowed=False,
            reason="空命令",
            rejected_command=command,
        )

    # 检测空字节（安全攻击常用）
    if "\x00" in command:
        return ValidationResult(
            is_allowed=False,
            reason="命令中包含空字节，不允许执行",
            rejected_command=command,
        )

    # 检测整条命令的注释
    if command.startswith("#"):
        return ValidationResult(
            is_allowed=False,
            reason="注释不是有效命令",
            rejected_command=command,
        )

    # 先检测整条命令的禁止模式
    ok, reason = _check_rejected_patterns(command)
    if not ok:
        logger.warning("命令被拒绝（禁止模式）: %s, 原因: %s", command, reason)
        return ValidationResult(
            is_allowed=False,
            reason=reason,
            rejected_command=command,
        )

    # 进入单命令递归校验
    result = _validate_single_command(command, allowed_script_dirs)

    if result.is_allowed:
        logger.debug("命令通过校验: %s", command)
    else:
        logger.warning(
            "命令被拒绝: %s, 原因: %s, 被拒命令: %s",
            command,
            result.reason,
            result.rejected_command or command,
        )

    return result


def is_command_allowed(
    command: str,
    *,
    allowed_script_dirs: list[str] | None = None,
    strict: bool = True,
) -> bool:
    """快捷方法：返回命令是否允许执行。

    Args:
        command: 待校验的命令字符串。
        allowed_script_dirs: 允许执行脚本的目录列表。
        strict: 严格模式。

    Returns:
        命令是否允许。

    Example:
        >>> is_command_allowed("ls /tmp")
        True
        >>> is_command_allowed("rm file.txt")
        False
    """
    result = validate_command(
        command,
        allowed_script_dirs=allowed_script_dirs,
        strict=strict,
    )
    return result.is_allowed


# ========== 导出 ==========

__all__ = [
    "ValidationResult",
    "ParameterRestriction",
    "AllowedFlagsOnly",
    "ForbiddenFlags",
    "ALLOWED_COMMANDS",
    "PARAMETER_RESTRICTIONS",
    "EFFECTIVE_ALLOWED_COMMANDS",
    "DEFAULT_ALLOWED_SCRIPT_DIRS",
    "validate_command",
    "is_command_allowed",
    "validate_path",
    "redact_output",
    "_normalize_command_name",
]
