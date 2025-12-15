/**
 * useIframeDragResize - iframe 拖拽调整大小处理 Composable
 * 
 * 职责：
 * - 禁用/启用页面中所有 iframe 的指针事件
 * - 包装拖拽和调整大小事件，解决 iframe 导致的事件丢失问题
 */
import { ref, Ref } from 'vue';

export interface UseIframeDragResizeOptions {
  /** 拖拽处理函数 */
  handleDragging: (x: number, y: number) => void;
  /** 调整大小处理函数 */
  handleResizing: (x: number, y: number, width: number, height: number) => void;
  /** 拖拽停止处理函数 */
  handleDragStop: (x: number, y: number) => void;
  /** 调整大小停止处理函数 */
  handleResizeStop: (x: number, y: number, width: number, height: number) => void;
}

export interface UseIframeDragResizeReturn {
  /** 是否正在拖拽或调整大小 */
  isDraggingOrResizing: Ref<boolean>;
  /** 禁用 iframe 指针事件 */
  disableIframePointerEvents: () => void;
  /** 启用 iframe 指针事件 */
  enableIframePointerEvents: () => void;
  /** 拖拽时的处理（包含 iframe 禁用） */
  handleDraggingWithIframe: (x: number, y: number) => void;
  /** 调整大小时的处理（包含 iframe 禁用） */
  handleResizingWithIframe: (x: number, y: number, width: number, height: number) => void;
  /** 拖拽停止时的处理（包含 iframe 恢复） */
  handleDragStopWithIframe: (x: number, y: number) => void;
  /** 调整大小停止时的处理（包含 iframe 恢复） */
  handleResizeStopWithIframe: (x: number, y: number, width: number, height: number) => void;
}

export function useIframeDragResize(options: UseIframeDragResizeOptions): UseIframeDragResizeReturn {
  const {
    handleDragging,
    handleResizing,
    handleDragStop,
    handleResizeStop,
  } = options;

  const isDraggingOrResizing = ref(false);

  /**
   * 禁用页面中所有 iframe 的指针事件
   * 用于解决拖拽时鼠标经过 iframe 导致事件丢失的问题
   */
  const disableIframePointerEvents = (): void => {
    const iframes = document.querySelectorAll('iframe');
    iframes.forEach(iframe => {
      iframe.style.pointerEvents = 'none';
    });
  };

  /**
   * 恢复页面中所有 iframe 的指针事件
   */
  const enableIframePointerEvents = (): void => {
    const iframes = document.querySelectorAll('iframe');
    iframes.forEach(iframe => {
      iframe.style.pointerEvents = '';
    });
  };

  /**
   * 拖拽时的处理（包含 iframe 禁用）
   */
  const handleDraggingWithIframe = (x: number, y: number): void => {
    // 首次拖拽时禁用 iframe 的指针事件
    if (!isDraggingOrResizing.value) {
      isDraggingOrResizing.value = true;
      disableIframePointerEvents();
    }
    handleDragging(x, y);
  };

  /**
   * 调整大小时的处理（包含 iframe 禁用）
   */
  const handleResizingWithIframe = (x: number, y: number, width: number, height: number): void => {
    // 首次调整大小时禁用 iframe 的指针事件
    if (!isDraggingOrResizing.value) {
      isDraggingOrResizing.value = true;
      disableIframePointerEvents();
    }
    handleResizing(x, y, width, height);
  };

  /**
   * 拖拽停止时的处理（包含 iframe 恢复）
   */
  const handleDragStopWithIframe = (x: number, y: number): void => {
    handleDragStop(x, y);
    isDraggingOrResizing.value = false;
    enableIframePointerEvents();
  };

  /**
   * 调整大小停止时的处理（包含 iframe 恢复）
   */
  const handleResizeStopWithIframe = (x: number, y: number, width: number, height: number): void => {
    handleResizeStop(x, y, width, height);
    isDraggingOrResizing.value = false;
    enableIframePointerEvents();
  };

  return {
    isDraggingOrResizing,
    disableIframePointerEvents,
    enableIframePointerEvents,
    handleDraggingWithIframe,
    handleResizingWithIframe,
    handleDragStopWithIframe,
    handleResizeStopWithIframe,
  };
}
