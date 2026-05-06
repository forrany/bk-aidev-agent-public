# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from dataclasses import dataclass

from aidev_agent.config import settings
from aidev_agent.core.nodes.model.pydantic_models import NextFunction, ProcessorContext
from aidev_agent.core.tools.skill.registry import SkillRegistry
from aidev_agent.core.tools.skill.types import SkillOptions
from aidev_agent.packages.resource_manager.registry import resource_manager


def _extract_local_params(skill: SkillOptions, config: dict) -> dict:
    """提取 FilesystemBackend 的构造参数。

    FilesystemBackend.__init__ 签名：
        root_dir=None, virtual_mode=False, max_file_size_mb=10, envs=None

    返回空字典，使用 FilesystemBackend 默认参数。
    """
    return {}


def _extract_e2b_params(skill: SkillOptions, config: dict) -> dict:
    """提取 E2BSandboxBackend 的构造参数。

    E2BSandboxBackend.__init__ 签名：
        template="sdt-hcomwqox", timeout=600, api_key=None, domain=None, envs=None

    返回空字典，使用 E2BSandboxBackend 默认参数。
    api_key/domain 从环境变量读取（使用默认值）。
    """
    return {}


def _extract_paas_params(skill: SkillOptions, config: dict) -> dict:
    """提取 PaasSandboxBackend 的构造参数。

    PaasSandboxBackend.__init__ 签名（全部 keyword-only）：
        app_code: str = "",
        access_token: str = "",
        snapshot: str,
        snapshot_entrypoint: list[str],
        env_vars: dict,

    参数来源优先级：
        - access_token: resource_manager().resolve_access_token(username) > 环境变量 SANDBOX_BP_ACCESS_TOKEN
        - snapshot / env_vars: 从 skill metadata 的 bkai_paas_sandbox 字段获取
        - app_code: 从 settings 获取

    env_vars 构建逻辑已委托给 ``resource_manager().build_skill_env()``。
    """
    access_token = resource_manager().resolve_access_token(config.get("executor")) or os.getenv(
        "SANDBOX_BP_ACCESS_TOKEN", ""
    )

    # 从 skill metadata 中获取 paas sandbox 配置
    paas_sandbox = {}
    if skill:
        paas_sandbox = skill.get("metadata", {}).get("bkai_paas_sandbox", {})

    # 委托 resource_manager 构建 env_vars（传入 username 作为 fallback）
    env_vars = resource_manager().build_skill_env(skill_config=skill, username=config.get("executor"))

    return {
        "app_code": settings.APP_CODE,
        "bk_username": config.get("executor"),
        "access_token": access_token,
        "snapshot": paas_sandbox.get("image", ""),
        "snapshot_entrypoint": [],
        "env_vars": env_vars,
    }


@dataclass
class SkillsPromptMiddleware:
    """将 Agent Skills 发现信息注入到 system prompt 中（模板管道）。"""

    registry: SkillRegistry
    skill_sources: list[str]
    enable_runtime_tool: bool = False

    def __call__(self, ctx: ProcessorContext, next: NextFunction) -> None:
        summary = self.registry.get_skills_summary(include_runtime=self.enable_runtime_tool)

        parts: list[str] = [
            "\n\n",
            "# Agent Skills\n",
            "\n技能（Skill）是 Agent 能力的外部延伸，遵循**“按需感知，受控触发”**的渐进式披露原则。每一个技能描述都包含其核心逻辑、环境约束与触发边界。",
            "\n1. 技能的核心定义",
            "\n功能封装：技能是针对特定任务（如数据提取、系统操作、协议转换）的原子化工具集。",
            "\n环境隔离：每个技能运行在独立的 Runtime 环境中，具有特定的 Shell 限制和生命周期。",
            "\n数据旁路：高性能技能通常采用“非对称”设计，即数据流直接在后端处理或下载，不强制经过模型 Context，以确保零 Token 消耗。",
            "\n2. 感知与触发机制",
            "\nAgent 必须通过扫描用户指令中的**“意图特征”**来感知识别技能：",
            "\n场景匹配：当用户需求命中技能描述中的 使用场景 (Scenario) 列表时，该技能的加载优先级高于一切通用推理逻辑。",
            "\n隐式推理：若用户请求的任务规模（如“海量”、“全量”）或技术栈（如特定协议、API）暗示了技能的必要性，应优先选择技能路径。",
            "\n3. 使用规范",
            "\n一旦识别到匹配技能，Agent 必须严格遵守以下 “激活-执行” 协议：",
            "\n主动激活(必须): 在任何实质性操作前，必须首先调用 activate_skill(skill_name=...)。",
            "\n环境适配(必须): 执行技能相关命令时，请将技能的 Runtime 值作为 `runtime` 参数传入`execute` 等工具。"
            "\n路径：skill 脚本通常在 ~/.agents/skills/<skill_name> 中，如果特定 runtime 有指定路径，以指定为主，注意, 不要猜测 ~ 的指向，应该直接使用 ~ 符号",
            "\n参数必须从用户上下文提取，严禁凭空编造。",
            f"\n\n可用技能列表：\n{summary}\n\n",
        ]

        if self.enable_runtime_tool:
            parts.append(
                "[注意：无状态沙箱环境] 对于 paas_sandbox 该环境是非持久化的"
                "临时性：每次用户发送消息，你都将面对一个完全初始化的环境"
                "禁止依赖：禁止假设之前的执行结果（如临时文件、后台进程、环境变量）在当前轮次依然可用。"
                "原子化执行：请尽可能将“准备、执行、清理”逻辑集成在单次指令中，以确保任务的连续性。\n"
                "[注意：合并指令] 对于 paas_sandbox 命令是单次的。"
                "错误做法：先执行 ls，再对话，再执行 grep"
                "正确做法：在一次 execute 中使用 && 或管道符，例如：ls -l | grep 'target'"
                "[注意：路径约束] 执行命令时的 Working Directory 即为技能根目录，初始内容是从 ～/.agents/skills/<skill_name>/ 复制的。直接运行脚本，不要去修改脚本，如果脚本不能执行可以分析原因，但不可以绕过脚本直接编码执行"
            )

        injection = "".join(parts)
        ctx.prompt_slots.system = (ctx.prompt_slots.system or "") + injection
        next()
