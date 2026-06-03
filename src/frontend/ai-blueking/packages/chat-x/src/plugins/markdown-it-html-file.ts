/*
 * Tencent is pleased to support the open source community by making
 * 蓝鲸智云PaaS平台 (BlueKing PaaS) available.
 *
 * Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
 *
 * 蓝鲸智云PaaS平台 (BlueKing PaaS) is licensed under the MIT License.
 *
 * License for 蓝鲸智云PaaS平台 (BlueKing PaaS):
 *
 * ---------------------------------------------------
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of
 * the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

import type { Options, PluginSimple, Renderer, Token } from '../markdown-it';

/** fence 渲染规则类型 */
type FenceRenderRule = NonNullable<Renderer['rules']['fence']>;

/**
 * 解析 html-file 代码块的 meta 行
 * 支持格式：
 *   filename="dashboard.html"
 *   {"filename": "dashboard.html"}
 */
function parseMetaLine(line: string): { filename?: string } {
  const trimmed = line.trim();

  // JSON 格式: {"filename": "xxx"}
  if (trimmed.startsWith('{')) {
    try {
      const obj = JSON.parse(trimmed) as Record<string, unknown>;
      return typeof obj.filename === 'string' ? { filename: obj.filename } : {};
    } catch {
      return {};
    }
  }

  // key=value 格式: filename="xxx"
  const match = trimmed.match(/filename=["']?([^"'\s]+)["']?/);
  return match?.[1] ? { filename: match[1] } : {};
}

/**
 * 检查是否为 html-file 代码块
 */
const isHtmlFileBlock = (info: string): boolean => {
  return info === 'html-file' || info.startsWith('html-file ');
};

/**
 * Markdown-it 插件：将 ```html-file 代码块渲染为占位 div
 * 与 mermaid 插件模式一致：返回 placeholder div + data 属性
 * MarkdownContent 通过 token.info 检测并渲染 Vue 组件
 *
 * 语法：
 * ```html-file filename="dashboard.html"
 * <!DOCTYPE html>
 * <html>...</html>
 * ```
 */
export const markdownItHtmlFile: PluginSimple = md => {
  const defaultFenceRenderer = md.renderer.rules.fence as FenceRenderRule;

  md.renderer.rules.fence = (tokens: Token[], idx: number, options: Options, env: unknown, self: Renderer): string => {
    const token = tokens[idx]!;
    const info = token.info ? md.utils.unescapeAll(token.info).trim() : '';

    if (isHtmlFileBlock(info)) {
      const rawContent = token.content || '';
      const lines = rawContent.split('\n');

      // 第一行是 meta
      const meta = parseMetaLine(lines[0] ?? '');
      const htmlContent = lines.slice(1).join('\n').trim();

      const encodedFilename = encodeURIComponent(meta.filename ?? 'untitled.html');
      const encodedContent = encodeURIComponent(htmlContent);

      // 占位 div，数据编码在属性中（与 mermaid 模式一致）
      return `<div class="html-file-card-wrapper" data-filename="${encodedFilename}" data-content="${encodedContent}" data-token-idx="${idx}"></div>`;
    }

    return defaultFenceRenderer(tokens, idx, options, env, self);
  };
};

export default markdownItHtmlFile;
