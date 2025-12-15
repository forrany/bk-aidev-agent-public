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
 * THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

import { t } from '../lang';
import type { AiBluekingProps, DropdownMenuConfig } from '../types/ai-blueking-props';
import type { IShortcut, IRequestOptions } from '../types';

/**
 * AI Blueking 组件 Props 默认值
 * 使用函数形式返回默认值，避免引用类型共享问题
 */
export const aiBluekingPropsDefaults = {
  title: '',
  extCls: '',
  helloText: () => t('你好，我是小鲸'),
  enablePopup: true,
  shortcuts: (): IShortcut[] => [],
  shortcutLimit: 3,
  shortcutFilter: undefined,
  hideDefaultTrigger: false,
  url: '',
  prompts: (): string[] => [],
  hideNimbus: false,
  requestOptions: (): IRequestOptions => ({}),
  defaultMinimize: false,
  teleportTo: 'body',
  draggable: true,
  defaultWidth: undefined,
  defaultHeight: undefined,
  defaultTop: undefined,
  defaultLeft: undefined,
  hideHeader: false,
  disabledInput: false,
  nimbusSize: 'normal' as const,
  showHistoryIcon: true,
  showNewChatIcon: true,
  showCompressionIcon: true,
  showMoreIcon: true,
  placeholder: () => t('输入 "/" 唤出 Prompt\n通过 Shift + Enter 进行换行输入'),
  miniPadding: 0,
  initialSessionCode: '',
  autoSwitchToInitialSession: false,
  loadRecentSessionOnMount: false,
  dropdownMenuConfig: (): DropdownMenuConfig => ({
    showRename: true,
    showAutoGenerate: true,
    showShare: true,
  }),
  defaultChatInputPosition: undefined,
  maxWidth: 1000,
} as const satisfies {
  [K in keyof Required<AiBluekingProps>]: AiBluekingProps[K] | (() => AiBluekingProps[K]);
};

/**
 * 获取 Props 默认值的类型安全辅助函数
 * 用于 withDefaults 中使用
 */
export type PropsDefaults = typeof aiBluekingPropsDefaults;
