<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { getRuntimeGlobal } from '../utils/runtime-globals'

interface BackToAidevProps {
  /** screen：移动端抽屉内展示为整行按钮 */
  variant?: 'nav' | 'screen'
}

withDefaults(defineProps<BackToAidevProps>(), {
  variant: 'nav',
})

/** 运行时由 VitePress dev / npm 包 / Express 注入 window.BK_AIDEV_URL，无值时不渲染 */
const aidevUrl = ref('')

onMounted(() => {
  aidevUrl.value = getRuntimeGlobal('BK_AIDEV_URL').trim()
})
</script>

<template>
  <div
    v-if="aidevUrl"
    class="back-to-aidev"
    :class="`is-${variant}`"
  >
    <a
      class="back-to-aidev__btn"
      :href="aidevUrl"
    >
      返回 AIDev
    </a>
  </div>
</template>

<style scoped>
.back-to-aidev.is-nav {
  display: none;
  align-items: center;
  margin-left: 12px;
}

@media (min-width: 768px) {
  .back-to-aidev.is-nav {
    display: flex;
  }
}

.back-to-aidev.is-screen {
  display: block;
  padding: 12px 24px 24px;
}

.back-to-aidev__btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 32px;
  padding: 0 14px;
  border: 1px solid var(--vp-button-brand-border);
  border-radius: 20px;
  background-color: var(--vp-button-brand-bg);
  color: var(--vp-button-brand-text);
  font-size: 13px;
  font-weight: 600;
  line-height: 1;
  white-space: nowrap;
  text-decoration: none;
  transition:
    color 0.25s,
    border-color 0.25s,
    background-color 0.25s;
}

.back-to-aidev__btn:hover {
  border-color: var(--vp-button-brand-hover-border);
  background-color: var(--vp-button-brand-hover-bg);
  color: var(--vp-button-brand-hover-text);
}

.back-to-aidev__btn:active {
  border-color: var(--vp-button-brand-active-border);
  background-color: var(--vp-button-brand-active-bg);
  color: var(--vp-button-brand-active-text);
}

.back-to-aidev.is-screen .back-to-aidev__btn {
  width: 100%;
  height: 40px;
  font-size: 14px;
}
</style>
