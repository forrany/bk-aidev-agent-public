# -*- coding: utf-8 -*-
"""Activate skill tool.

This tool returns the full SKILL.md body (without YAML frontmatter) for a given
skill name.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from .registry import SkillRegistry


def get_activate_skill_tool(registry: SkillRegistry) -> BaseTool:
    def activate_skill(skill_name: str) -> str:
        """Load and return full instructions for a skill."""

        skill = registry.activate_skill(skill_name)
        if skill is None:
            available = ", ".join([s["name"] for s in registry.list_skills()])
            return f"Error: Unknown skill '{skill_name}'. Available skills: {available or '(none)'}"
        return skill.get("instructions", "") or ""

    return StructuredTool.from_function(
        name="activate_skill",
        description="Activate a skill by name and return its full SKILL.md instructions (YAML frontmatter removed).",
        func=activate_skill,
    )
