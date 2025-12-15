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

import type { IRequestOptions, IShortcut } from './index';

/**
 * AI Blueking 组件下拉菜单配置
 */
export interface DropdownMenuConfig {
  showRename?: boolean;
  showAutoGenerate?: boolean;
  showShare?: boolean;
}

/**
 * AI Blueking 组件 Props 类型定义
 */
export interface AiBluekingProps {
  /** 额外的 CSS 类名 */
  extCls?: string;

  /** 标题 */
  title?: string;

  /** 问候语标题 */
  helloText?: string;

  /** 是否启用弹出框功能 */
  enablePopup?: boolean;

  /** 快捷指令列表 */
  shortcuts?: IShortcut[];

  /** 快捷指令显示数量限制 */
  shortcutLimit?: number;

  /** 快捷指令过滤函数 */
  shortcutFilter?: (shortcut: IShortcut, selectedText: string) => boolean;

  /** 是否隐藏默认触发器 */
  hideDefaultTrigger?: boolean;

  /** API 请求地址 */
  url?: string;

  /** 预设提示词列表 */
  prompts?: string[];

  /** 是否隐藏 Nimbus 悬浮球 */
  hideNimbus?: boolean;

  /** 请求配置选项 */
  requestOptions?: IRequestOptions;

  /** 默认是否最小化 */
  defaultMinimize?: boolean;

  /** Teleport 目标元素选择器 */
  teleportTo?: string;

  /** 是否可拖拽 */
  draggable?: boolean;

  /** 默认宽度 */
  defaultWidth?: number;

  /** 默认高度 */
  defaultHeight?: number;

  /** 默认顶部位置 */
  defaultTop?: number;

  /** 默认左侧位置 */
  defaultLeft?: number;

  /** 是否隐藏头部 */
  hideHeader?: boolean;

  /** 是否禁用输入 */
  disabledInput?: boolean;

  /** Nimbus 悬浮球大小 */
  nimbusSize?: 'small' | 'normal' | 'large';

  /** 是否显示历史记录图标 */
  showHistoryIcon?: boolean;

  /** 是否显示新建对话图标 */
  showNewChatIcon?: boolean;

  /** 输入框占位符 */
  placeholder?: string;

  /** 最小内边距 */
  miniPadding?: number;

  /** 初始会话代码 */
  initialSessionCode?: string;

  /** 是否自动切换到初始会话 */
  autoSwitchToInitialSession?: boolean;

  /** 挂载时是否加载最近会话 */
  loadRecentSessionOnMount?: boolean;

  /** 下拉菜单配置 */
  dropdownMenuConfig?: DropdownMenuConfig;

  /**
   * 是否显示缩放图标
   * @since v1.2.9
   * @default true
   * @description 如果未定义，则显示压缩图标
   */
  showCompressionIcon?: boolean;

  /**
   * 是否显示更多图标
   * @since v1.2.9
   * @default true
   * @description 如果未定义，则显示更多图标
   */
  showMoreIcon?: boolean;

  /**
   * 默认输入框位置
   * @since v1.2.9
   * @default undefined
   * @description 如果未定义，则根据是否有会话内容自动判断位置
   * @example
   * - undefined: 根据是否有会话内容自动判断位置
   * - 'bottom': 底部
   */
  defaultChatInputPosition?: 'bottom' | undefined;

  /**
   * 最大宽度
   * @since v1.2.9
   * @default undefined
   * @description 如果未定义，1000px
   * @example
   * - undefined: 1000px
   * - 100: 最大宽度为 100px
   * - 100%: 最大宽度为视窗宽度
   */
  maxWidth?: number | string;
}

/**
 * 位置信息类型
 */
export interface PositionInfo {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * AI Blueking 组件 Emits 类型定义
 */
export interface AiBluekingEmits {
  (e: 'shortcut-click', data: { shortcut: IShortcut; source: 'popup' | 'main' }): void;
  (e: 'close' | 'show' | 'stop' | 'receive-start' | 'receive-text' | 'receive-end'): void;
  (e: 'send-message', message: string): void;
  (e: 'session-initialized', data: { openingRemark: string; predefinedQuestions: string[] }): void;
  (e: 'sdk-error', data: { apiName: string; code: number; message: string; data: unknown }): void;
  (e: 'transfer-messages', messageIds: string[]): void;
  (e: 'share-messages', messageIds: string[]): void;
  (e: 'dragging', position: PositionInfo): void;
  (e: 'resizing', position: PositionInfo): void;
  (e: 'drag-stop', position: PositionInfo): void;
  (e: 'resize-stop', position: PositionInfo): void;
}
