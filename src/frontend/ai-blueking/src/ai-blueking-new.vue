<template>
  <teleport :to="teleportTo">
    <div
      :class="['ai-blueking-wrapper', props.extCls ?? '']"
      :style="rootVariables"
    >
      <vue-draggable-resizable
        v-if="isShow"
        :active="isShow"
        :draggable="props.draggable"
        :h="height"
        :max-width="maxWidth"
        :min-height="minHeight"
        :min-width="minWidth"
        :parent="true"
        :prevent-deactivation="true"
        :resizable="true"
        :w="width"
        :x="left"
        :y="top"
        class-name="ai-blueking-container-wrapper"
        drag-handle=".drag-handle"
        @dragging="handleDraggingWithIframe"
        @resizing="handleResizingWithIframe"
        @drag-stop="handleDragStopWithIframe"
        @resize-stop="handleResizeStopWithIframe"
      >
        <div
          ref="rootNode"
          class="ai-blueking-container"
        >
          <loading-overlay :show="isLoadingSessionContents" />
          <!-- 顶部栏 -->
          <ai-blueking-header
            v-if="!props.hideHeader"
            :title="props.title"
            :is-compression-height="isCompressionHeight"
            :draggable="props.draggable"
            :show-history-icon="props.showHistoryIcon"
            :show-new-chat-icon="props.showNewChatIcon"
            :enable-chat-session="enableChatSession"
            :chat-group="sessionStore.agentInfo.value?.chatGroup"
            :has-session-contents="hasSessionContents"
            :dropdown-menu-config="props.dropdownMenuConfig"
            :show-compression-icon="props.showCompressionIcon"
            @close="handleClose"
            @toggle-compression="toggleCompression"
            @new-chat="handleNewChat"
            @auto-generate-name="handleAutoGenerateName"
            @help-click="() => enterSelectMode('transfer')"
          />
          <div class="content-wrapper">
            <!-- 无权限提示 -->
            <div
              v-if="!hasPermission"
              class="permission-denied"
            >
              <bk-exception
                class="exception-wrap-item exception-part"
                title="暂无该智能体使用权限"
                description="请联系 admin（管理员） 处理"
                scene="part"
                type="403"
              ></bk-exception>
            </div>
            <!-- 主要内容区域 -->
            <div
              v-else
              :class="`main-content ${!hasSessionContents ? 'greeting-layout' : 'chat-layout'}`"
            >
              <greeting-section
                ref="greetingSectionRef"
                :title="props.helloText"
                :greeting-text="greetingText"
                :has-session-contents="hasSessionContents"
                :render-markdown="renderMarkdown"
                :greeting-max-height="greetingMaxHeight"
              />
              <message-list
                ref="messageListRef"
                :session-contents="sessionContents"
                :has-session-contents="hasSessionContents"
                :content-margin-bottom="contentMarginBottom"
                :is-select-mode="sessionStore.isSelectMode.value"
                :is-message-selected="sessionStore.isMessageSelected"
                @delete="handleDelete"
                @regenerate="handleRegenerate"
                @resend="handleResend"
                @message-select="sessionStore.toggleMessageSelection"
                @update-session-content="handleUpdateSessionContent"
                @scroll-position-change="handleScrollPositionChange"
              />
              <!-- 选择模式下的底部确认区域 -->
              <div
                v-if="sessionStore.isSelectMode.value"
                class="selection-footer"
              >
                <div class="selection-info">
                  <bk-checkbox
                    :model-value="isSelectAll"
                    :indeterminate="isIndeterminate"
                    label="全选"
                    @change="handleSelectAllChange"
                  />
                </div>
                <div class="selection-actions">
                  <bk-button
                    class="cancel-btn"
                    @click="handleCancelSelection"
                  >
                    {{ t('取消') }}
                  </bk-button>
                  <bk-button
                    class="confirm-btn"
                    :loading="loading"
                    theme="primary"
                    @click="handleConfirmSelection"
                  >
                    {{ t('确定') }}
                  </bk-button>
                </div>
              </div>

              <motion.div
                v-else
                :transition="{
                  duration: 0.5,
                  ease: [0.33, 1, 0.68, 1],
                  type: 'tween',
                  layoutId: 'chat-input',
                }"
                :class="`chat-input-container ${!hasSessionContents ? (props.defaultChatInputPosition ?? 'centered') : 'bottom'}`"
                :style="
                  hasSessionContents || props.defaultChatInputPosition
                    ? undefined
                    : inputContainerStyle
                "
                :layout="!hasSessionContents"
              >
                <div class="chat-input-wrapper">
                  <div
                    v-if="currentSessionLoading || showScrollToBottom"
                    class="bottom-tools-bar"
                  >
                    <bar-button
                      v-if="currentSessionLoading"
                      color="#EA3636"
                      icon="bkai-icon bkai-tingzhishengcheng"
                      :text="t('停止生成')"
                      @click="handleStop"
                    />
                    <bar-button
                      v-if="showScrollToBottom"
                      color="#979BA5"
                      icon="bkai-icon bkai-jiantou"
                      :text="t('返回底部')"
                      @click="handleScrollMainToBottom"
                    />
                  </div>
                  <custom-input
                    v-if="mergedShortcut"
                    :key="mergedShortcut.bindKey"
                    :shortcut="mergedShortcut"
                    :root-node="rootNode"
                    @cancel="handleCancelShortcut"
                    @submit="handleSubmitShortcut"
                  />

                  <chat-input-box
                    v-else
                    v-model="inputMessage"
                    :class="!hasSessionContents ? 'greeting-layout' : 'chat-layout'"
                    :loading="currentSessionLoading || false"
                    :prompts="promptList"
                    :shortcuts="props.shortcuts"
                    :shortcut-filter="props.shortcutFilter"
                    :conversation-settings="sessionStore.agentInfo.value?.conversationSettings"
                    :disabled="props.disabledInput"
                    :placeholder="props.placeholder"
                    @height-change="handleInputHeightChange"
                    @send="handleSendMessage"
                    @shortcut-click="handleInputShortcutClick"
                    @stop="handleStop"
                  />
                </div>
              </motion.div>
            </div>
          </div>
        </div>
      </vue-draggable-resizable>
      <nimbus
        v-if="!props.hideNimbus"
        v-model:is-panel-show="isShow"
        v-model:is-minimize="isNimbusMinimize"
        :default-minimize="defaultMinimize"
        :size="props.nimbusSize"
        @click="handleNimbusClick"
      />
      <render-popup
        :shortcuts="props.shortcuts"
        :conversation-settings="sessionStore.agentInfo.value?.conversationSettings"
        :shortcut-limit="props.shortcutLimit"
        :shortcut-filter="props.shortcutFilter"
        :hide-default-trigger="props.hideDefaultTrigger"
        @click="isShow = true"
        @shortcut-click="handlePopupShortcutClick"
      />
    </div>
  </teleport>
</template>

<script setup lang="ts">
  // ===================================================================
  // 1. 导入和类型定义
  // ===================================================================
  import { useChat, useStyle, useClickProxy } from '@blueking/ai-ui-sdk/hooks';
  import { Button as BkButton, Checkbox as BkCheckbox, Exception as BkException } from 'bkui-vue';
  import { motion } from 'motion-v';
  import { computed, nextTick, provide, ref, Ref, watch } from 'vue';
  import VueDraggableResizable from 'vue-draggable-resizable';

  // 组件导入
  import AiBluekingHeader from './components/ai-header.vue';
  import BarButton from './components/bar-button.vue';
  import ChatInputBox from './components/chat-input-box.vue';
  import CustomInput from './components/custom-inputs/index.vue';
  import GreetingSection from './components/greeting-section.vue';
  import LoadingOverlay from './components/loading-overlay.vue';
  import MessageList from './components/message-list.vue';
  import RenderPopup from './components/render-popup.vue';
  // Composable导入
  import { useChatCore } from './composables/use-chat-core';
  import { useGreetingHeight } from './composables/use-greeting-height';
  import { useIframeDragResize } from './composables/use-iframe-drag-resize';
  import { useKeyboardShortcut } from './composables/use-keyboard-shortcut';
  import { useMarkdown } from './composables/use-markdown';
  import { useMessageList } from './composables/use-message-list';
  import { POPUP_INJECTION_KEY } from './composables/use-popup-props';
  import { useResizableContainer } from './composables/use-resizable-container';
  import { useSelect } from './composables/use-select-pop';
  import { useSelectionMode } from './composables/use-selection-mode';
  import { provideSessionStore } from './composables/use-session-store';
  import { useShortcut } from './composables/use-shortcut';
  // 配置和工具导入
  import { aiBluekingPropsDefaults } from './config/props-defaults';
  import { t } from './lang';
  import type { AiBluekingProps, AiBluekingEmits } from './types/ai-blueking-props';
  import type { IShortcut } from './types';
  import { normalizeUrl } from './utils';
  import Nimbus from './views/nimbus.vue';

  // 样式导入
  import 'vue-draggable-resizable/style.css';

  // ===================================================================
  // 2. Props 和 Emits 定义
  // ===================================================================
  const props = withDefaults(defineProps<AiBluekingProps>(), aiBluekingPropsDefaults);

  const emit = defineEmits<AiBluekingEmits>();

  // ===================================================================
  // 3. 依赖注入
  // ===================================================================
  provide(POPUP_INJECTION_KEY, props.enablePopup);

  // 标准化的URL，自动匹配当前页面协议
  const normalizedUrl = computed(() => {
    return normalizeUrl(props.url);
  });

  // 提供给子组件使用
  provide('normalizedUrl', normalizedUrl);

  // ===================================================================
  // 4. 核心状态管理
  // ===================================================================
  // DOM引用
  const chatInputBoxRef = ref<InstanceType<typeof ChatInputBox>>();
  const rootNode: Ref<HTMLElement | undefined> = ref();

  // UI状态
  const isShow = ref(false);
  const isNimbusMinimize = ref(props.defaultMinimize);
  const inputMessage = ref('');

  // 输入状态
  const inputHeight = ref(68);

  // ===================================================================
  // 5. 会话存储
  // ===================================================================
  const sessionStore = provideSessionStore();

  // ===================================================================
  // 6. 组合式 API 钩子 (Composables)
  // ===================================================================
  // 使用 markdown 渲染功能
  const { renderMarkdown } = useMarkdown();

  // 使用选择文本功能
  const { selectedText, citeText, setCiteText } = useSelect(props.enablePopup);

  // 使用消息列表功能
  const { messageListRef, scrollMainToBottom, scrollToBottomIfNeeded, resetUserScrolling } =
    useMessageList();

  // 使用快捷方式功能
  const { currentShortcut, handleShortcutClick, handleCancelShortcut } = useShortcut({
    selectedText: selectedText,
    isShow: isShow,
    handleShow: () => {
      handleShow();
    },
    handleStop: () => {
      handleStop();
    },
  });

  // 动态合并 currentShortcut 和原始 shortcuts 的最新数据
  // 当 props.shortcuts 变化时，自动更新 options 等动态属性
  const mergedShortcut = computed(() => {
    if (!currentShortcut.value) return undefined;

    // 从原始 shortcuts 中找到对应的 shortcut
    const originalShortcut = props.shortcuts?.find(s => s.id === currentShortcut.value?.id);
    if (!originalShortcut) return currentShortcut.value;

    // 合并 components，保留 currentShortcut 中的运行时属性（如 selectedText），更新 options 等配置属性
    const mergedComponents = currentShortcut.value.components?.map((comp, index) => {
      const originalComp = originalShortcut.components?.[index];
      if (originalComp && comp.key === originalComp.key) {
        // 合并：currentShortcut 的运行时属性 + originalShortcut 的最新配置
        return {
          ...originalComp, // 最新的配置（options、placeholder 等）
          selectedText: (comp as any).selectedText, // 保留运行时添加的 selectedText
        };
      }
      return comp;
    });

    return {
      ...currentShortcut.value,
      components: mergedComponents,
      bindKey: currentShortcut.value.id + '_' + Date.now(),
    };
  });

  /**
   * 计算根元素的样式变量
   * @since v1.2.9
   * @description 计算根元素的样式变量
   * @example
   * - maxWidth: 1000px
   * - maxWidth: 100%
   * @returns {Record<string, string>}
   * @example
   * {
   *   '--ai-blueking-max-width': '1000px',
   * }
   */
  const rootVariables = computed(() => {
    const maxWidth = typeof props.maxWidth === 'number' ? `${props.maxWidth}px` : props.maxWidth;
    return {
      '--ai-blueking-max-width': maxWidth ?? '1000px',
    };
  });

  // 初始化样式和点击代理
  useStyle();
  useClickProxy();

  // ===================================================================
  // 7. UI 和布局管理 (UI & Layout)
  // ===================================================================
  // 使用可调整大小的容器
  const {
    minWidth,
    minHeight,
    maxWidth,
    top,
    left,
    width,
    height,
    isCompressionHeight,
    handleDragging,
    handleResizing,
    handleDragStop,
    handleResizeStop,
    toggleCompression,
    updatePosition,
    updateSize,
    updatePositionAndSize,
  } = useResizableContainer(
    {
      maxWidthPercent: 80,
      initWidth: props.defaultWidth,
      defaultHeight: props.defaultHeight,
      defaultTop: props.defaultTop,
      defaultLeft: props.defaultLeft,
      miniPadding: props.miniPadding,
    },
    position => {
      // 拖拽结束回调
      emit('drag-stop', position);
    },
    position => {
      // 调整大小结束回调
      emit('resize-stop', position);
    },
    position => {
      // 拖拽过程中回调
      emit('dragging', position);
    },
    position => {
      // 调整大小过程中回调
      emit('resizing', position);
    }
  );

  // 使用 iframe 拖拽调整大小处理
  const {
    handleDraggingWithIframe,
    handleResizingWithIframe,
    handleDragStopWithIframe,
    handleResizeStopWithIframe,
  } = useIframeDragResize({
    handleDragging,
    handleResizing,
    handleDragStop,
    handleResizeStop,
  });

  // 动态计算 greeting 最大高度
  const windowHeightForGreeting = ref(window.innerHeight);
  const greetingMaxHeight = computed(() => windowHeightForGreeting.value - 367);

  // 使用问候语高度计算功能
  const { updateGreetingTextHeight, getInputContainerStyle } = useGreetingHeight({
    greetingMaxHeight: greetingMaxHeight.value,
  });

  // 计算输入框的动态位置
  const inputContainerStyle = computed(() => {
    return getInputContainerStyle(hasSessionContents.value);
  });

  // 内容边距计算
  const contentMarginBottom = computed(() => {
    const toolsBarHeight = 40;
    const offSetHeight = selectedText.value ? 100 : 70;
    return inputHeight.value + toolsBarHeight + offSetHeight;
  });

  // ===================================================================
  // 8. 聊天和会话管理 (Chat & Session)
  // ===================================================================

  // 使用聊天功能
  const {
    currentSession,
    sessionContents,
    isLoadingSessionContents,
    plusSessionContent,
    plusSessionApi,
    chat,
    stopChat,
    setCurrentSession,
    setCurrentSessionChain,
    currentSessionLoading,
    reGenerateChat,
    reSendChat,
    deleteChat,
    updateRequestOptions,
    getAgentInfoApi,
    getChatGroupApi,
    getSessionsApi,
    getSessionContentsApi,
    modifySessionApi,
    deleteSessionApi,
    renameSessionApi,
    setSessionContents,
    handleCompleteRole,
    stopSessionContentApi,
    shareSessionApi,
  } = useChat({
    handleStart: () => {
      scrollToBottomIfNeeded();
      emit('receive-start');
    },
    handleText: () => {
      scrollToBottomIfNeeded();
      emit('receive-text');
    },
    handleEnd: () => {
      scrollToBottomIfNeeded();
      emit('receive-end');
    },
    requestOptions: {
      url: normalizedUrl.value,
      ...props.requestOptions,
    },
    otherOptions: {
      hideReferenceDocIcon: true,
    },
  });

  // 使用选择模式功能
  const {
    handleConfirmSelection,
    handleCancelSelection,
    handleSelectAllChange,
    enterSelectMode,
    isSelectAll,
    isIndeterminate,
    loading,
  } = useSelectionMode({
    sessionStore,
    sessionContents,
    getChatGroupApi,
    shareSessionApi,
    onTransferMessages: messageIds => {
      emit('transfer-messages', messageIds);
    },
    onShareMessages: messageIds => {
      emit('share-messages', messageIds);
    },
  });

  // 注册 SDK 的方法
  sessionStore.registerSdkMethods({
    setCurrentSession,
    setCurrentSessionChain,
    setSessionContents,
    modifySessionApi,
    deleteSessionApi,
    renameSessionApi,
    getSessionContentsApi,
    getSessionsApi,
    getAgentInfoApi,
    plusSessionApi,
    handleCompleteRole,
    getChatGroupApi,
    shareSessionApi,
  });

  // 注册错误回调函数
  sessionStore.registerErrorCallback(error => {
    emit('sdk-error', error);
  });

  // ===================================================================
  // 9. 核心聊天逻辑 (使用 useChatCore 整合)
  // ===================================================================
  const {
    // 状态
    showScrollToBottom,
    // 计算属性
    promptList,
    hasSessionContents,
    greetingText,
    // 会话管理方法
    switchToSession,
    handleNewChat,
    handleAutoGenerateName,
    // 消息操作方法
    handleSendMessage,
    handleRegenerate,
    handleResend,
    handleStop,
    handleSubmitShortcut,
    handleDelete,
    handleUpdateSessionContent,
    // UI 事件处理
    handleShow,
    handleClose,
    handleScrollMainToBottom,
    handleScrollPositionChange,
  } = useChatCore({
    normalizedUrl,
    sessionStore,
    props: {
      loadRecentSessionOnMount: props.loadRecentSessionOnMount,
      initialSessionCode: props.initialSessionCode,
      autoSwitchToInitialSession: props.autoSwitchToInitialSession,
      prompts: props.prompts,
      requestOptions: props.requestOptions,
    },
    chatMethods: {
      currentSession,
      sessionContents,
      currentSessionLoading,
      plusSessionContent: plusSessionContent as any,
      chat: chat as any,
      stopChat: stopChat as any,
      reGenerateChat: reGenerateChat as any,
      reSendChat: reSendChat as any,
      deleteChat: deleteChat as any,
      renameSessionApi,
      stopSessionContentApi,
      setSessionContents,
      updateRequestOptions,
    },
    messageListMethods: {
      scrollMainToBottom,
      scrollToBottomIfNeeded,
      resetUserScrolling,
      messageListRef: messageListRef as any,
    },
    selectMethods: {
      citeText,
      setCiteText,
    },
    shortcutMethods: {
      handleCancelShortcut,
    },
    selectionModeMethods: {
      enterSelectMode,
    },
    greetingMethods: {
      updateGreetingTextHeight,
    },
    inputMessage,
    isShow,
    emit: (e: any, data?: any) => emit(e, data),
  });

  // 是否启用会话管理
  const enableChatSession = computed(() => {
    return sessionStore.agentInfo.value?.conversationSettings?.enableChatSession ?? true;
  });

  // 是否有权限
  const hasPermission = computed(() => {
    return sessionStore.hasPermission.value;
  });

  // ===================================================================
  // 10. Watcher 监听器 (isShow 变化时聚焦输入框)
  // ===================================================================
  watch(
    () => isShow.value,
    newValue => {
      if (newValue) {
        nextTick(() => {
          chatInputBoxRef.value?.focus();
        });
      }
    }
  );

  // ===================================================================
  // 11. 事件处理函数
  // ===================================================================

  const handlePopupShortcutClick = (data: { shortcut: IShortcut; source: 'popup' | 'main' }) => {
    // 来自 render-popup 的快捷方式点击事件
    emit('shortcut-click', { shortcut: data.shortcut, source: data.source });
    handleShortcutClick(data);
  };

  const handleInputShortcutClick = (data: { shortcut: IShortcut; source: 'popup' | 'main' }) => {
    // 传递_source属性以便custom-input组件可以正确处理自动提交
    const modifiedShortcut = {
      ...data.shortcut,
      _source: data.source,
    };
    emit('shortcut-click', { shortcut: modifiedShortcut, source: data.source });
    handleShortcutClick({ shortcut: modifiedShortcut, source: data.source });
  };

  const handleInputHeightChange = (height: number) => {
    inputHeight.value = height;
  };

  // ===================================================================
  // 12. 快捷键管理
  // ===================================================================
  useKeyboardShortcut({
    enabled: computed(() => !props.hideNimbus),
    isShow,
    onShow: handleShow,
    onHide: handleClose,
  });

  const handleNimbusClick = () => {
    handleShow();
  };

  // ===================================================================
  // 13. 暴露给父组件的方法
  // ===================================================================
  defineExpose({
    stopSessionContentApi,
    sessionContents,
    handleShow,
    handleClose,
    handleStop,
    handleSendMessage,
    handleShortcutClick,
    handleDelete,
    handleRegenerate,
    handleResend,
    updateRequestOptions,
    currentSessionLoading,
    isLoadingSessionContents,
    updateGreetingTextHeight,
    setCurrentSession,
    setCiteText,
    focusInput: () => {
      chatInputBoxRef.value?.focus();
    },
    // 新增会话接口
    addNewSession: (sessionCode?: string) => sessionStore.addNewSession(sessionCode),
    // 更新会话名称接口
    updateSessionName: async (sessionCode: string, newName: string) => {
      return sessionStore.updateSession(sessionCode, { sessionName: newName });
    },
    // 切换到指定会话
    switchToSession,
    // 获取会话列表
    getSessionList: sessionStore.getSessionList,
    sessionList: sessionStore.sessionList,
    // 是否启用会话管理
    enableChatSession,
    // 编程式控制容器位置和大小
    updatePosition,
    updateSize,
    updatePositionAndSize,
  });
</script>

<style lang="scss" scoped>
  @use './styles/mixins.scss';
  @use './styles/markdown.scss';

  .ai-blueking-wrapper {
    position: fixed;
    top: 0;
    left: 0;
    z-index: 10000;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
  }

  .ai-blueking-container-wrapper {
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100%;
    pointer-events: auto;
    background: #ffffff;
    border-radius: 2px;
    box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.2);

    :deep(.handle) {
      background: transparent;
      border: none;

      &.handle-ml,
      &.handle-mr {
        top: 0;
        height: 100%;
        margin-top: 0;
        cursor: ew-resize;
      }

      &.handle-tm,
      &.handle-bm {
        left: 0;
        width: 100%;
        margin-left: 0;
        cursor: ns-resize;
      }

      &.handle-tl,
      &.handle-br {
        cursor: nwse-resize;
      }

      &.handle-tr {
        top: -5px;
        right: -5px;
      }

      &.handle-tl {
        top: -5px;
        left: -5px;
      }

      &.handle-bl {
        bottom: -5px;
        left: -5px;
      }

      &.handle-br {
        right: -5px;
        bottom: -5px;
      }

      &.handle-tr,
      &.handle-bl {
        cursor: nesw-resize;
      }
    }
  }

  .ai-blueking-container {
    position: relative;
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100%;
    :deep(.icon-search::before) {
      content: none !important;
    }
  }

  .content-wrapper {
    position: relative;
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  .main-content {
    position: relative;
    display: flex;
    flex: 1;
    flex-direction: column;
    min-width: 0;
    height: 100%;
    padding: 16px;
    overflow-y: auto;

    .bottom-tools-bar {
      display: flex;
      gap: 8px;
      align-items: center;
      justify-content: center;
      margin-bottom: 16px;
    }

    .message-line-wrapper {
      width: 100%;
      max-width: var(--ai-blueking-max-width);
      margin: 0 auto;
    }

    &.chat-layout {
      gap: 0;
    }

    .chat-input-wrapper {
      max-width: var(--ai-blueking-max-width);
      margin: 0 auto;
    }

    .chat-input-container {
      z-index: 3;
      width: 100%;
      outline: none;
      transform-origin: center;
      will-change: transform;

      &.centered {
        position: absolute;
        top: 188px;
        left: 50%;
        width: calc(100% - 32px);

        /* stylelint-disable-next-line declaration-no-important */
        transform: translate(-50%, 0) !important;
      }

      &.bottom {
        position: absolute;
        right: 16px;
        bottom: 16px;
        left: 16px;
        width: calc(100% - 32px);
      }
    }

    .selection-footer {
      position: absolute;
      bottom: 0;
      left: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
      height: 46px;
      padding: 0 16px;
      background: #fafbfd;
      box-shadow: inset 0 1px 0 0 #dcdee5;

      .selection-info {
        font-size: 14px;
        color: #63656e;
      }

      .selection-actions {
        display: flex;
        gap: 8px;

        .cancel-btn {
          color: #63656e;
          background: #ffffff;
          border: 1px solid #c4c6cc;

          &:hover {
            border-color: #979ba5;
          }
        }
      }
    }
  }

  .action-icon {
    cursor: pointer;
    opacity: 0.7;

    &:hover {
      opacity: 1;
    }
  }

  // 添加 Nimbus 相关样式
  :deep(.vdr) {
    background: transparent;
    border: none;
  }

  // 无权限提示样式
  .permission-denied {
    display: flex;
    margin-top: 140px;
    justify-content: center;
    flex: 1;
    height: 100%;

    .permission-denied-content {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 40px 20px;
      text-align: center;

      .permission-icon {
        font-size: 48px;
        color: #c4c6cc;
        margin-bottom: 16px;
      }

      .permission-text {
        font-size: 18px;
        font-weight: 600;
        color: #63656e;
        margin-bottom: 8px;
      }

      .permission-desc {
        font-size: 14px;
        color: #979ba5;
      }
    }
  }
</style>
