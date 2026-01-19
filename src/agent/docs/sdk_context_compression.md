# SDK 侧上下文压缩与超限保护（三层机制）

本文整理 SDK（aidev-agent）在 **上下文超限** 场景下的处理机制与推荐用法，核心目标是：

- 在不明显损失关键信息的前提下，降低进入模型的上下文体积；
- 给工具调用结果提供“软压缩 + 硬兜底”的双重保护；
- 在最终构造 prompt 时，对知识库内容与聊天历史做 token 超限治理。

> 重点：这是 **分层防线**，越靠前越“定向、可控”，越靠后越“通用兜底”。

---

## 0. 三层策略总览

| 层级 | 发生位置 | 处理对象 | 手段 | 主要收益 |
|---|---|---|---|---|
| 1 | **Tool 级（按工具定制）** | 单个工具返回 | 执行后处理（压缩/格式化/结构化提取/转 Command） | 最精准，能保留业务关键字段 |
| 2 | **ToolNode 级（通用拦截）** | 所有工具返回（ToolMessage） | 超长硬截断（替换为一行提示） | 防止意外超大结果“瞬间打爆”上下文 |
| 3 | **Prompt 级（最终兜底）** | 工具消息/知识库内容/聊天历史 | LLM 摘要 + token 超限检测与渐进式压缩 | 保证模型调用前上下文可控 |

---

## 1. 第一层：LangGraph Tool 增强（EnhancedTool）

### 1.1 核心思路

对“容易产出大结果”的工具，建议在 SDK 侧 **显式包一层 EnhancedTool**，在原工具执行完成后对结果做二次处理，从源头减少进入上下文的数据量。

实现见：`aidev_agent/packages/langchain_core/tools/enhance.py`。

EnhancedTool 的关键特性：

1. **可为工具参数 schema 增加必填字段 `invoke_intent`**（调用意图）
   - 对 Pydantic schema 会动态 create_model 扩展字段（`enhance.py:61`）
   - 对 MCP 导出的 dict schema 也会注入 properties/required（`enhance.py:80`）
2. **执行原工具后调用 `compressor_func`** 对结果做压缩（`enhance.py:197` / `enhance.py:217`）
3. **压缩失败可降级** 输出原始结果（`fallback_on_error=True`，`enhance.py:284`）
4. 同步/异步均支持。

### 1.2 `invoke_intent` 的价值

当 `show_intent=True` 时，EnhancedTool 会强制要求参数包含 `invoke_intent`（`enhance.py:170`），其设计目的不是“给模型看的装饰字段”，而是：

- 让压缩函数能基于 **本次调用目的** 做“相关性裁剪”（只保留与意图有关的字段/片段）；
- 给后续的 LLM 压缩、规则压缩提供更可靠的语义线索；
- 降低“工具输出很大，但真正有用信息很少”的典型浪费。

### 1.3 `compressor_func` 形态与建议

协议定义：

- `CompressorFunc.__call__(original_result, tool_name, invoke_intent=..., **kwargs) -> str`（`enhance.py:41`）

常用实现建议（从轻到重）：

1. **规则压缩**：提取必要字段、删掉大字段、分页/截断、JSON 选择性保留。
2. **基于意图的裁剪**：根据 `invoke_intent` 只保留相关 section。
3. **LLM 压缩**：将工具结果交给小模型/同模型摘要（注意成本与延迟）。
4. **转 Command（可选）**：将“原始大结果”放入 state/metadata（中间态），返回小摘要或报告。LangGraph 原生支持工具返回 `Command` 更新状态；EnhancedTool 本身 `_run`/`_arun` 返回 `Any`（`enhance.py:197`），因此压缩函数也可按需返回 `Command`（需确保上层 ToolNode 能接住）。

创建工具的便利函数：`create_enhanced_tool(...)`（`enhance.py:306`）。

### 1.4 示例（包装一个易爆工具）

```python
from aidev_agent.packages.langchain_core.tools.enhance import create_enhanced_tool


def compressor(original_result, tool_name: str, *, invoke_intent: str | None = None, **kwargs) -> str:
    # 例：规则压缩（仅示意）
    # - 保留关键字段
    # - 删除大字段（如 raw_log / html / base64 等）
    # - 结合 invoke_intent 做相关性筛选
    return summarize_result(original_result, intent=invoke_intent)


enhanced = create_enhanced_tool(
    original_tool=my_tool,
    compressor_func=compressor,
    show_intent=True,  # 强制 LLM 提供 invoke_intent
    fallback_on_error=True,
)
```

---

## 2. 第二层：ToolNode 通用拦截（result_limit_wrapper）

### 2.1 核心思路

即使做了第一层增强，仍可能遇到：

- 新接入的工具未做压缩；
- 工具返回异常（误输出超大 payload）；
- LLM 生成了不合理参数导致工具全量导出。

因此 ToolNode 层提供 **通用硬兜底**：若工具返回的 `ToolMessage.content` 长度超过阈值，直接替换为一行提示，避免上下文被瞬间塞爆。

- 拦截实现：`aidev_agent/core/nodes/tool/result_limit_wrapper.py:18`
- ToolNode 构造与 wrapper 链：`aidev_agent/core/nodes/tool/node.py:158`

### 2.2 行为细节

- 仅对 `ToolMessage` 生效；`Command` 不做替换（`result_limit_wrapper.py:25`）。
- 判断依据是 `len(str(msg.content))`（字符长度），不是 token 计数（`result_limit_wrapper.py:14`）。
- 超限替换文案：`本次工具调用返回结果超长，请重新调整调用参数`（`result_limit_wrapper.py:11`）。

### 2.3 如何启用

`ToolNodeSettings` 默认 **不开启** result limit（`settings.py:6`）：

- `use_result_limit: bool = False`
- `result_limit_thrd: int = 1000`

在 `build_tool_node(...)` 中，若 `node_options.use_result_limit=True` 会自动挂载 wrapper：

- 同步：`build_result_limit_sync_wrapper(node_options.result_limit_thrd)`（`node.py:216`）
- 异步：`build_result_limit_async_wrapper(...)`（`node.py:218`）

示例：

```python
from aidev_agent.core.nodes.tool import build_tool_node
from aidev_agent.core.nodes.tool.settings import ToolNodeSettings

node = build_tool_node(
    tools=[...],
    handle_tool_errors=True,
    node_options=ToolNodeSettings(use_result_limit=True, result_limit_thrd=5000),
)
```

> 建议：第二层阈值可以略大于第三层的 ToolOutputCompressionMiddleware 阈值（见下文），用于兜住“极端爆炸”。

---

## 3. 第三层：整体结果 token_compression（最终避免超限）

第三层发生在 **模型调用前的 prompt 变量构造阶段**，由 `ContextProcessor` 的 variable pipeline 驱动（`context_processor.py:148`）。

关键点：第三层关注的是“**最终进入模型的上下文**”，因此会同时处理：

- 工具消息（tool_messages）
- 知识库召回内容（context）
- 聊天历史（chat_history）

### 3.1 工具输出压缩：ToolOutputCompressionMiddleware

实现：`aidev_agent/core/nodes/model/token_compression.py:311`，在 pipeline 中最先执行（`context_processor.py:151`）。

行为：

- 从 `ctx.metadata["tool_messages"]` 读取工具消息（由 `BaseVariablesMiddleware` 切分得到，见 `variables.py:90`）。
- 若所有 ToolMessage 的 content 拼接长度 `> tool_output_compress_thrd`，触发压缩（`token_compression.py:446`）。
- 对每条 ToolMessage 并行调用 LLM 摘要（线程池），回写到新的 ToolMessage 列表（`token_compression.py:465` / `token_compression.py:492`）。
- 压缩模式：
  - `compressor_type="common"`：通用摘要（`token_compression.py:402`）
  - `compressor_type="specific"`：带“与用户提问相关性判断”的摘要（`token_compression.py:406`）

默认阈值来自环境变量 `TOOL_OUTPUT_COMPRESS_THRD`，ContextProcessor 默认 5000（`context_processor.py:110`）。

> 与第二层区别：
> - 第二层是“硬替换为一行提示”。
> - 第三层是“LLM 摘要保留关键信息”，更适合正常业务。

### 3.2 知识库内容压缩：KnowledgeCompressionMiddleware

实现：`token_compression.py:141`，在 variable pipeline 中执行（`context_processor.py:166`）。

行为：

- 前提：必须提供 `ctx.llm`、`ctx.chat_prompt_template`、`ctx.token_limit`，且 `ctx.variables["context"]` 非空（`token_compression.py:152`）。
- token 超限判定：使用 `llm.get_num_tokens_from_messages(...)` 计算（`token_compression.py:115`），阈值为 `token_limit - token_margin`（`token_compression.py:135`）。
- 压缩调用：依赖 `intent_recognition_instance.llm_context_compressor_parallel(...)`（`token_compression.py:183` / `token_compression.py:193`）。
- **哈希缓存复用**：对知识库内容做 sha256，内容不变时在 ReAct 循环中复用上次压缩结果（`token_compression.py:163` / `token_compression.py:171`）。

### 3.3 聊天历史压缩：ChatHistoryCompressionMiddleware

实现：`token_compression.py:210`，在 variable pipeline 中最后执行（`context_processor.py:167`）。

行为：

- 仅在 token 超限时触发。
- 采用“**渐进式移除最早消息**”的策略，直到不超限或无可移除（`token_compression.py:246`）。
- 会在 `_compression_state` 中累计 `chat_history_removed`，确保跨 ReAct 循环可复用（`token_compression.py:218` / `token_compression.py:262`）。
- 特别注意：该 middleware 强调“不修改原始 state.messages”，而是改写 `ctx.variables["chat_history"]`（`token_compression.py:211` / `token_compression.py:260`）。

### 3.4 可观测性：压缩日志事件

BaseCompressionMiddleware 会通过 `conditional_dispatch_custom_event` 发送压缩日志（`token_compression.py:106`），事件数据形如：

- `{"compress_log": "..."}`（`token_compression.py:110`）

这有助于在 SDK/前端侧展示“发生了压缩、压缩了什么”。

---

## 4. 推荐配置与实践建议

### 4.1 启用顺序建议

1. **优先做第一层（EnhancedTool）**：对已知大输出工具“定向治理”。
2. **打开第二层（ToolNode result limit）**：作为线上硬兜底，防止极端情况。
3. **依赖第三层（token_compression）**：保证最终 prompt 不超限；同时适配知识库与聊天历史。

### 4.2 阈值调优建议

- `ToolNodeSettings.result_limit_thrd`（第二层）
  - 倾向设置为一个“异常保护值”（例如 5k~20k 字符），避免误伤正常工具。
- `TOOL_OUTPUT_COMPRESS_THRD`（第三层工具摘要）
  - 倾向设置为一个“正常业务压缩点”（例如 3k~10k 字符），让摘要在进入 token 级压缩前就发生。
- `token_limit` / `token_margin`（第三层 token 超限判断）
  - `token_margin` 建议保留一定冗余（默认 100，见 `context_processor.py:235`），避免边界情况下因 token 估算误差导致模型调用失败。

### 4.3 工具设计建议（从源头减少压缩成本）

- 工具 API 支持分页、筛选字段、时间窗口、topK 等参数；
- 默认返回“摘要 + 可选 detail”，不要默认全量 dump；
- 对日志/列表类工具，优先返回：
  - 关键统计（数量、分布、错误码 topN）
  - 代表性样本（前 N 条）
  - 可复现的查询条件

---

## 5. 排查清单（出现上下文超限/输出异常时）

1. 是否某个工具返回了超大 payload？
   - 开启第二层后，应该会变成一行提示（`result_limit_wrapper.py:11`）。
2. 是否 tool_messages 在第三层被摘要了？
   - 检查是否收到 `compress_log` 事件（`token_compression.py:108`）。
3. 是否 token_limit 未正确传入导致第三层未生效？
   - `KnowledgeCompressionMiddleware` / `ChatHistoryCompressionMiddleware` 都要求 `ctx.token_limit` 存在（`token_compression.py:153` / `token_compression.py:214`）。
4. 是否知识库压缩缺少 `intent_recognition_instance`？
   - 缺失会跳过知识库压缩（`token_compression.py:183`）。
