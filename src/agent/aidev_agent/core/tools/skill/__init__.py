# -*- coding: utf-8 -*-
"""aidev_agent.core.tools.skill

Skills infrastructure: metadata types, discovery/loading, registry, and activation tool.

This package provides:
- skill metadata parsing and discovery (local_provider)
- registry for lazy loading + activation (registry)
- activate_skill tool for LLM agents (activate_skill)
- SkillProvider protocol for extending skill sources (types)
- LocalSkillProvider for local filesystem skills (local_provider)
"""

from .activate_skill import get_activate_skill_tool
from .local_provider import LocalSkillProvider
from .registry import SkillRegistry
from .types import SkillOptions, SkillProvider

__all__ = [
    "LocalSkillProvider",
    "SkillOptions",
    "SkillProvider",
    "SkillRegistry",
    "get_activate_skill_tool",
]
