---
name: /doc-update
description: 启动功能开发后文档与日志更新流程
position: 1
---

当在 Claude Code 中执行 `/doc-update` 命令时，将启动功能开发后的文档与日志更新流程。

## 使用方式

1. 基于指定commit范围更新：
   `/doc-update --from <commit-hash> --to <commit-hash>`

2. 基于最近的commit更新：
   `/doc-update --last <number-of-commits>`

3. 基于分支差异更新：
   `/doc-update --branch <branch-name>`

4. 基于自定义描述更新：
   `/doc-update [功能名称或简要描述]`

例如：
- `/doc-update --last 3`
- `/doc-update --from abc1234 --to def5678`
- `/doc-update --branch feature/new-session-api`
- `/doc-update 检查下 git 记录中 1.3.1 相关的变更`

## 执行流程

### 阶段一：Commit 分析与变更识别

1. 根据参数分析 git commit 记录
2. 识别变更类型：新功能、优化改进、Bug 修复、破坏性变更
3. 整理变更内容摘要

### 阶段二：文档更新检查清单

检查以下文档是否需要更新：

#### 必须检查的文档

| 文档位置 | 用途 | 更新时机 |
|---------|------|---------|
| `ai-blueking/CHANGELOG.md` | NPM 包更新日志，开发者查看 | 每个版本发布时必须更新 |
| `web/docs/changelog.md` | 用户文档站更新日志 | 每个版本发布时必须更新 |
| `ai-blueking/readme.md` | NPM 包 README，API 参考 | 新增/变更 Props、Events、Methods 时更新 |

#### 按变更类型检查

| 变更类型 | 需要更新的文档 |
|---------|---------------|
| 新增 Props | `ai-blueking/readme.md` 属性表格、`ai-blueking/src/types/ai-blueking-props.ts` |
| 新增 Events | `ai-blueking/readme.md` 事件表格 |
| 新增 Methods | `ai-blueking/readme.md` 方法表格、`ai-blueking/src/types/index.ts` (AIBluekingExpose) |
| 快捷操作变更 | `web/docs/guide/core-features/shortcuts.md` |
| 会话管理变更 | `web/docs/guide/advanced-usage/session-lifecycle.md` |
| 编程式交互变更 | `web/docs/guide/advanced-usage/programmatic-interaction.md` |
| 请求配置变更 | `web/docs/guide/advanced-usage/custom-requests.md` |
| 破坏性变更 | 所有相关文档 + 迁移指南 |

### 阶段三：文档内容实现

1. **CHANGELOG.md 格式**：
   ```markdown
   ## [版本号] - YYYY-MM-DD

   ### ✨ 新增功能
   - **功能名称**：功能描述

   ### 🛠️ 优化改进
   - **优化名称**：优化描述

   ### 🐛 BUG修复
   - 修复问题描述
   ```

2. **readme.md 更新原则**：
   - 确保所有示例代码使用正确的接口（如 `IShortcut` 而非旧的 `ShortCut`）
   - Props/Events/Methods 表格与代码类型定义保持一致
   - 高级用法示例应该是可运行的完整代码

3. **web/docs/changelog.md 格式**：
   ```javascript
   {
     version: "v1.x.x",
     date: "YYYY-MM-DD",
     important: "⚠️ 重要提醒（如有）",
     features: ["新功能列表"],
     improvements: ["优化改进列表"],
     fixes: ["Bug修复列表"],
     breaking: ["破坏性变更列表（如有）"]
   }
   ```

### 阶段四：验证与一致性检查

1. **版本号一致性**：检查 `package.json` 版本号与文档中版本号是否匹配
2. **类型定义一致性**：检查 readme.md 中的接口定义是否与 `src/types/` 中的实际类型一致
3. **示例代码正确性**：确保示例代码使用的方法和属性名称正确
4. **链接有效性**：检查文档中的内部链接是否有效

## 注意事项

1. **不要遗漏 CHANGELOG.md**：`ai-blueking/CHANGELOG.md` 是给 NPM 包用户看的，`web/docs/changelog.md` 是文档站展示用的，两者都需要更新

2. **readme.md 容易过时**：`ai-blueking/readme.md` 经常被遗漏更新，特别注意检查：
   - 版本号徽章是否最新
   - 特性列表是否完整
   - 属性/事件/方法表格是否与代码同步
   - 示例代码是否使用最新的 API

3. **快捷操作接口已变更**：v1.1.0 起使用 `IShortcut` 接口（包含 `components` 配置），不再使用旧的 `ShortCut` 接口（包含 `prompt` 模板）

4. **检查 Vue2 同步**：如果有新增 Props 或 Methods，检查 `vue2.ts` 是否同步暴露

接下来，请根据：$ARGUMENTS 内容，更新相关文档
