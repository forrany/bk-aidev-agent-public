---
name: /spec
description: 启动 Kiro Spec 工作流，生成需求、设计和任务规划文档
position: 0
---

当在 Claude Code 中执行 `/spec` 命令时，将启动 Kiro Spec 工作流，该工作流包含以下三个阶段：

1. 需求收集 (Requirements Gathering)
2. 设计文档创建 (Design Document Creation)
3. 实现规划 (Implementation Planning)

使用方法：
/spec [功能名称或简要描述]

例如：
/spec 编程式会话管理
/spec 新增用户偏好设置功能

执行该命令后，Claude Code 将自动调用 spec 子代理来处理完整的规范流程。