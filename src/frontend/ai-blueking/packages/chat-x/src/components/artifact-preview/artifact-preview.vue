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
        <span class="artifact-preview-title">{{ title }}</span>
      </div>
      <div class="artifact-preview-actions">
        <slot name="locateButton" />
        <!-- 复制 -->
        <div
          v-tippy="{ content: '复制', theme: 'ai-chat-box' }"
          class="artifact-action-btn"
          @click="handleCopy"
        >
          <svg
            class="ai-common-icon"
            style="vertical-align: middle; fill: currentColor; overflow: hidden"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path d="M672 704L672 832C672 867.346224 643.346224 896 608 896L192 896C156.653776 896 128 867.346224 128 832L128 384C128 348.653776 156.653776 320 192 320L352 320 352 216.88888896C352 168.2420416 389.7269152 128 437.3333344 128L810.6666656 128C858.2730848 128 896 168.2420416 896 216.88888896L896 615.1111104C896 663.7579584 858.2730848 704 810.6666656 704L672 704ZM608 384L192 384 192 832 608 832 608 384ZM672 640L810.6666656 640C821.9706208 640 832 629.3019968 832 615.1111104L832 216.88888896C832 202.69800448 821.9706208 192 810.6666656 192L437.3333344 192C426.0293792 192 416 202.69800448 416 216.88888896L416 320 608 320C643.346224 320 672 348.653776 672 384L672 640Z" />
          </svg>
        </div>
        <!-- 新窗口打开 -->
        <div
          v-tippy="{ content: '在新窗口打开', theme: 'ai-chat-box' }"
          class="artifact-action-btn"
          @click="handleNewWindow"
        >
          <svg
            class="ai-common-icon"
            style="vertical-align: middle; fill: currentColor; overflow: hidden"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path d="M896 640L896 832C896 867.346 867.346 896 832 896L192 896C156.654 896 128 867.346 128 832L128 192C128 156.654 156.654 128 192 128L384 128L384 192L192 192L192 832L832 832L832 640L896 640ZM640 128L640 192L779.87 192L456.94 514.94L501.06 559.06L824 239.87L824 376L888 376L888 128L640 128Z" />
          </svg>
        </div>
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

  import { directive as vTippy } from 'vue-tippy';

  import { useArtifactPreviewConsumer } from '../../composables/use-artifact-preview';

  import 'tippy.js/dist/tippy.css';

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
    navigator.clipboard.writeText(content.value).catch(() => {
      // fallback: 用 textarea 复制
      const ta = document.createElement('textarea');
      ta.value = content.value;
      ta.style.position = 'fixed';
      ta.style.left = '-9999px';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    });
  };

  const handleNewWindow = () => {
    const blob = new Blob([content.value], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    window.open(url, '_blank');
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  };

  // 注意：不在 unmount 时删除 preview 数据，因为 tab 切换会触发 unmount
  // 数据由外部（如关闭 tab 时）显式清理
  onUnmounted(() => {
    window.removeEventListener('message', handleMessage);
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

  .artifact-preview-title {
    overflow: hidden;
    font-size: 13px;
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

  .artifact-action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    color: #979ba5;
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.15s;

    &:hover {
      color: #3a84ff;
      background: #f0f5ff;
    }

    .ai-common-icon {
      width: 14px;
      height: 14px;
    }
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
