import type { ISession } from '@blueking/ai-ui-sdk/types';

/**
 * 扩展 ISession 接口，添加 isEdit 属性
 */
export interface ISessionEditItem extends ISession {
  isEdit?: boolean;
}

/**
 * 会话历史分组项
 */
export interface HistoryItem {
  key: string;
  alias: string;
  sessionList: ISessionEditItem[];
} 