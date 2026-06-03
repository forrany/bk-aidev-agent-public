<!--
  Tencent is pleased to support the open source community by making
  蓝鲸智云PaaS平台 (BlueKing PaaS) available.

  Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.

  蓝鲸智云PaaS平台 (BlueKing PaaS) is licensed under the MIT License.

  License for 蓝鲸智云PaaS平台 (BlueKing PaaS):

  ---------------------------------------------------
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
  documentation files (the "Software"), to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
  to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
  the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
  THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
  CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
-->
<template>
  <div class="artifact-preview-container">
    <header class="artifact-preview-header">
      <div class="artifact-preview-header-left">
        <span class="artifact-preview-icon">🌐</span>
        <span class="artifact-preview-title">{{ title }}</span>
      </div>
      <div class="artifact-preview-actions">
        <slot name="locateButton" />
        <ToolBtn
          id="copy"
          description="复制 HTML"
          name="复制"
          @click="handleCopy"
        />
        <ToolBtn
          id="new-window"
          description="在新窗口打开"
          name="新窗口"
          @click="handleNewWindow"
        />
      </div>
    </header>
    <div class="artifact-preview-body">
      <iframe
        v-if="srcdocContent"
        ref="iframeRef"
        class="artifact-preview-iframe"
        sandbox="allow-scripts allow-forms"
        :srcdoc="srcdocContent"
      />
      <div
        v-else
        class="artifact-preview-empty"
      >
        等待内容...
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
  import { computed, nextTick, onUnmounted, ref, watch } from 'vue';

  import { useArtifactPreviewConsumer } from '../../composables/use-artifact-preview';
  import ToolBtn from '../ai-buttons/tool-btn/tool-btn.vue';

  const props = defineProps<{
    previewId: string;
  }>();

  const iframeRef = ref<HTMLIFrameElement>();
  const ctx = useArtifactPreviewConsumer();

  const entry = computed(() => ctx?.previews.value.get(props.previewId));
  const content = computed(() => entry.value?.content ?? '');
  const title = computed(() => entry.value?.filename ?? 'HTML 预览');

  /**
   * 在 srcdoc 末尾注入高度通知脚本
   * iframe 内通过 MutationObserver 监听 DOM 变化，实时通知父页面高度
   */
  const srcdocContent = computed(() => {
    const html = content.value;
    if (!html) return '';
    // 注入高度通知脚本
    const resizeScript = `<script>
(function(){
  function n(){try{window.parent.postMessage({type:'artifact-resize',id:'${props.previewId}',h:document.body.scrollHeight},'*')}catch(e){}}
  new MutationObserver(n).observe(document.body,{childList:true,subtree:true});
  window.addEventListener('resize',n);n();
})();
<\/script>`;
    // 在 </body> 前插入，如果没有 </body> 则追加到末尾
    if (html.includes('</body>')) {
      return html.replace('</body>', `${resizeScript}</body>`);
    }
    return html + resizeScript;
  });

  // iframe 高度自适应
  const autoResize = () => {
    const iframe = iframeRef.value;
    if (!iframe?.contentDocument?.body) return;
    const h = iframe.contentDocument.body.scrollHeight;
    if (h > 0) {
      iframe.style.height = `${Math.min(h + 20, 800)}px`;
    }
  };

  watch(srcdocContent, () => {
    nextTick(autoResize);
  });

  // 监听 iframe 内部的 postMessage 高度通知
  const handleMessage = (e: MessageEvent) => {
    if (e.data?.type === 'artifact-resize' && e.data.id === props.previewId) {
      if (iframeRef.value && e.data.h > 0) {
        iframeRef.value.style.height = `${Math.min(e.data.h + 20, 800)}px`;
      }
    }
  };
  window.addEventListener('message', handleMessage);

  const handleCopy = () => {
    navigator.clipboard.writeText(content.value);
  };

  const handleNewWindow = () => {
    const blob = new Blob([content.value], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    window.open(url, '_blank');
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  };

  onUnmounted(() => {
    window.removeEventListener('message', handleMessage);
    ctx?.removePreview(props.previewId);
  });
</script>

<style lang="scss">
  .artifact-preview-container {
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
    background: #f5f7fa;
  }

  .artifact-preview-header {
    display: flex;
    flex-shrink: 0;
    gap: 12px;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    background: #fff;
    border-bottom: 1px solid #dcdee5;
  }

  .artifact-preview-header-left {
    display: flex;
    gap: 8px;
    align-items: center;
    min-width: 0;
  }

  .artifact-preview-icon {
    flex-shrink: 0;
    font-size: 16px;
  }

  .artifact-preview-title {
    overflow: hidden;
    font-size: 14px;
    font-weight: 600;
    color: #313238;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .artifact-preview-actions {
    display: flex;
    flex-shrink: 0;
    gap: 4px;
    align-items: center;
    margin-left: 12px;
  }

  .artifact-preview-body {
    flex: 1;
    overflow: auto;
  }

  .artifact-preview-iframe {
    display: block;
    width: 100%;
    min-height: 200px;
    border: none;
  }

  .artifact-preview-empty {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 200px;
    color: #979ba5;
    font-size: 13px;
  }
</style>
