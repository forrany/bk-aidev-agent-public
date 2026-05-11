<template>
  <!-- 非 Assistant 消息：直接使用默认 MessageRender，不解析 custom-component -->
  <MessageRender
    v-if="message.role !== MessageRole.Assistant"
    :message="message"
  />

  <!-- Assistant 消息：解析 custom-component 并分发渲染 -->
  <div
    v-else
    class="custom-message-renderer"
  >
    <template
      v-for="(block, index) in blocks"
      :key="index"
    >
      <MessageRender
        v-if="block.type === 'text'"
        :message="{ ...message, content: block.content }"
      />
      <div
        v-else-if="block.type === 'custom'"
        class="custom-block-wrapper"
      >
        <ChartWidget
          v-if="block.data.type === 'chart'"
          :data="block.data"
        />
        <IframeWidget
          v-else-if="block.data.type === 'iframe'"
          :data="block.data"
        />
        <FormWidget
          v-else-if="block.data.type === 'form'"
          :data="block.data"
        />
        <div
          v-else
          class="unknown-block"
        >
          未知组件类型: {{ block.data.type }}
        </div>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
  import { computed } from 'vue';

  import { MessageRender, MessageRole } from '@blueking/chat-x';

  import { parseCustomBlocks } from '@blueking/ai-blueking';
  import ChartWidget from './ChartWidget.vue';
  import FormWidget from './FormWidget.vue';
  import IframeWidget from './IframeWidget.vue';

  import type { Message } from '@blueking/chat-x';

  const props = defineProps<{
    message: Message;
  }>();

  const blocks = computed(() => parseCustomBlocks(props.message.content || ''));
</script>

<style scoped>
  .custom-message-renderer {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .custom-block-wrapper {
    margin: 8px 0;
  }

  .unknown-block {
    padding: 12px;
    font-size: 13px;
    color: #ea3636;
    background: #fff5f5;
    border: 1px solid #fedddd;
    border-radius: 8px;
  }
</style>
