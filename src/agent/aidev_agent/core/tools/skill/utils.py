# -*- coding: utf-8 -*-

"""Shared utility functions for skill providers."""

from __future__ import annotations

import logging
import re
from typing import Optional

import yaml

from .types import SkillOptions

logger = logging.getLogger(__name__)


def parse_frontmatter(content: str) -> Optional[dict]:
    """Parse YAML frontmatter from skill markdown content.

    Parameters
    ----------
    content
        Raw markdown content that may start with ``---`` delimited YAML.

    Returns
    -------
    Optional[dict]
        Parsed frontmatter as a dict, or ``None`` if absent / invalid.
    """
    if not content.startswith("---"):
        return None

    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return None

    yaml_end = end_match.start() + 3
    yaml_content = content[3:yaml_end]

    try:
        frontmatter = yaml.safe_load(yaml_content)
        return frontmatter if isinstance(frontmatter, dict) else None
    except yaml.YAMLError as e:
        logger.warning(f"YAML 解析失败: {e}")
        return None


def extract_instructions(content: str) -> str:
    """Extract the instruction body from skill markdown, stripping frontmatter.

    Parameters
    ----------
    content
        Raw markdown content of a ``SKILL.md`` file.

    Returns
    -------
    str
        The body text with frontmatter removed and surrounding whitespace stripped.
    """
    if not content.startswith("---"):
        return content.strip()

    match = re.search(r"\n---\s*\n", content[3:])
    if match:
        start_pos = 3 + match.end()
        return content[start_pos:].strip()

    return content.strip()


def apply_optional_frontmatter_fields(
    target: SkillOptions,
    frontmatter: dict,
    *,
    stringify_simple_fields: bool = False,
    convert_list_elements: bool = False,
) -> None:
    """从 frontmatter 中提取可选字段并应用到目标字典。

    处理以下字段：license, compatibility, metadata, allowed-tools, runtime

    Parameters
    ----------
    target : dict
        目标字典（如 SkillOptions），会被原地修改
    frontmatter : dict
        从 YAML frontmatter 解析得到的字典
    stringify_simple_fields : bool, optional
        是否对 license 和 compatibility 字段强制转换为 str。
        默认为 False。设为 True 时行为与 bkai_provider 一致。
    convert_list_elements : bool, optional
        处理 allowed-tools 字段时，是否对 list 中的元素强制转换为 str。
        默认为 False。设为 True 时行为与 bkai_provider 一致。
    """
    # 处理 license
    if "license" in frontmatter:
        value = frontmatter["license"]
        target["license"] = str(value) if stringify_simple_fields else value

    # 处理 compatibility
    if "compatibility" in frontmatter:
        value = frontmatter["compatibility"]
        target["compatibility"] = str(value) if stringify_simple_fields else value

    # 处理 metadata（必须是 dict）
    if "metadata" in frontmatter and isinstance(frontmatter["metadata"], dict):
        target["metadata"] = frontmatter["metadata"]

    # 处理 allowed-tools（可以是 str 或 list）
    if "allowed-tools" in frontmatter:
        tools = frontmatter["allowed-tools"]
        if isinstance(tools, str):
            target["allowed_tools"] = tools.split()
        elif isinstance(tools, list):
            target["allowed_tools"] = [str(t) for t in tools] if convert_list_elements else tools

    # 处理 runtime（必须是 str）
    if "runtime" in frontmatter and isinstance(frontmatter["runtime"], str):
        target["runtime"] = frontmatter["runtime"]
