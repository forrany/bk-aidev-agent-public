<template>
  <div class="iframe-widget">
    <div class="iframe-header">
      <span class="iframe-icon">&#x1f310;</span>
      <span class="iframe-title">{{ data.title || '嵌入页面' }}</span>
      <a
        :href="data.src"
        target="_blank"
        class="iframe-open-link"
      >
        在新窗口打开
      </a>
    </div>
    <iframe
      :src="data.src"
      :style="{ height: (data.height || 400) + 'px' }"
      class="iframe-content"
      sandbox="allow-scripts allow-same-origin allow-forms"
      @load="handleLoad"
      @error="handleError"
    />
    <div
      v-if="loadError"
      class="iframe-error"
    >
      加载失败: {{ loadError }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

defineProps<{
  data: {
    title?: string;
    src: string;
    height?: number;
  };
}>();

const loadError = ref('');

const handleLoad = () => {
  loadError.value = '';
};

const handleError = () => {
  loadError.value = '无法加载嵌入内容';
};
</script>

<style scoped>
.iframe-widget {
  overflow: hidden;
  border: 1px solid #e1ecff;
  border-radius: 8px;
}

.iframe-header {
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 8px 12px;
  background: #f0f5ff;
  border-bottom: 1px solid #e1ecff;
}

.iframe-icon {
  font-size: 14px;
}

.iframe-title {
  flex: 1;
  font-size: 13px;
  font-weight: 500;
  color: #313238;
}

.iframe-open-link {
  font-size: 12px;
  color: #3a84ff;
  text-decoration: none;
}

.iframe-open-link:hover {
  text-decoration: underline;
}

.iframe-content {
  display: block;
  width: 100%;
  border: none;
}

.iframe-error {
  padding: 12px;
  font-size: 12px;
  color: #ea3636;
  text-align: center;
  background: #fff5f5;
}
</style>
