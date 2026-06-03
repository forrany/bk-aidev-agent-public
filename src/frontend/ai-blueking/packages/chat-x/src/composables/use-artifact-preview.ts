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

import { type InjectionKey, type ShallowRef, inject, provide, shallowRef } from 'vue';

export interface ArtifactPreviewEntry {
  /** artifact 类型：'html' | 'react' | 'vue' | 'svg' | ... */
  type: string;
  /** 渲染内容（HTML 源码等） */
  content: string;
  /** 可选文件名 */
  filename?: string;
}

export interface ArtifactPreviewContext {
  /** 所有活跃的预览条目 */
  previews: ShallowRef<Map<string, ArtifactPreviewEntry>>;
  /** 创建预览条目，返回唯一 id */
  createPreview: (type: string, content: string, meta?: { filename?: string }) => string;
  /** 更新预览内容（流式场景） */
  updatePreview: (id: string, content: string) => void;
  /** 移除预览条目 */
  removePreview: (id: string) => void;
}

export const ARTIFACT_PREVIEW_TOKEN: InjectionKey<ArtifactPreviewContext> = Symbol('ARTIFACT_PREVIEW');

let counter = 0;

/**
 * 在 ChatContainer 层调用，provide artifact preview 上下文给所有子孙组件
 */
export function useArtifactPreviewProvider(): ArtifactPreviewContext {
  const previews = shallowRef(new Map<string, ArtifactPreviewEntry>());

  const createPreview = (type: string, content: string, meta?: { filename?: string }): string => {
    const id = `artifact-${++counter}-${Date.now()}`;
    const newMap = new Map(previews.value);
    newMap.set(id, { type, content, filename: meta?.filename });
    previews.value = newMap;
    return id;
  };

  const updatePreview = (id: string, content: string): void => {
    const newMap = new Map(previews.value);
    const entry = newMap.get(id);
    if (entry) {
      newMap.set(id, { ...entry, content });
      previews.value = newMap;
    }
  };

  const removePreview = (id: string): void => {
    const newMap = new Map(previews.value);
    newMap.delete(id);
    previews.value = newMap;
  };

  const ctx: ArtifactPreviewContext = { previews, createPreview, updatePreview, removePreview };
  provide(ARTIFACT_PREVIEW_TOKEN, ctx);

  return ctx;
}

/**
 * 在子孙组件中调用，获取 artifact preview 上下文
 */
export function useArtifactPreviewConsumer(): ArtifactPreviewContext | undefined {
  return inject(ARTIFACT_PREVIEW_TOKEN);
}
