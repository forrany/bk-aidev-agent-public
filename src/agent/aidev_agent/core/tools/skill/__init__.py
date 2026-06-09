# -*- coding: utf-8 -*-
"""aidev_agent.core.tools.skill

Skills infrastructure: metadata types, discovery/loading, registry, and activation tool.

This package provides:
- skill metadata parsing and discovery (local_backend)
- registry for lazy loading + activation (provider)
- activate_skill tool for LLM agents (SkillRegistry method)
- SkillProviderBackend protocol for extending skill sources (types)
- LocalBackend for local filesystem skills (local_backend)
- BkAiBackend for BK-AIDev platform skills (bkai_backend)
"""

from .bkai_backend import BkAiBackend
from .local_backend import LocalBackend
from .provider import SkillRegistry
from .types import SkillOptions, SkillProviderBackend

__all__ = [
    "BkAiBackend",
    "LocalBackend",
    "SkillOptions",
    "SkillProviderBackend",
    "SkillRegistry",
]
