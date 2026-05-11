export interface CustomBlock {
  type: 'custom';
  data: Record<string, unknown>;
  raw: string;
}

export interface TextBlock {
  type: 'text';
  content: string;
}

export type ContentBlock = CustomBlock | TextBlock;

/**
 * 解析消息内容，将 ```custom-component 代码块识别为自定义组件，
 * 其余部分作为普通 Markdown 文本。
 *
 * AI prompt 示例：
 *   请使用以下格式输出自定义组件：
 *   ```custom-component
 *   {"type": "chart", "chartType": "bar", ...}
 *   ```
 */
export function parseCustomBlocks(content: unknown): ContentBlock[] {
  if (!content) return [];

  // content 可能不是 string（如 ReasoningMessage 的 string[]、ActivityMessage 的对象等）
  // 非 string 内容直接作为整体文本块返回，不做 custom-component 解析
  if (typeof content !== 'string') {
    return [{ type: 'text', content: content as string }];
  }

  const blocks: ContentBlock[] = [];
  const regex = /```custom-component\s*\n([\s\S]*?)\n```/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(content)) !== null) {
    // 前面的普通文本
    if (match.index > lastIndex) {
      const text = content.slice(lastIndex, match.index).trim();
      if (text) {
        blocks.push({ type: 'text', content: text });
      }
    }

    // 解析 JSON
    try {
      const data = JSON.parse(match[1]);
      blocks.push({ type: 'custom', data, raw: match[1] });
    } catch {
      // JSON 解析失败，当作普通文本
      blocks.push({ type: 'text', content: match[0] });
    }

    lastIndex = match.index + match[0].length;
  }

  // 剩余的普通文本
  if (lastIndex < content.length) {
    const text = content.slice(lastIndex).trim();
    if (text) {
      blocks.push({ type: 'text', content: text });
    }
  }

  return blocks;
}
