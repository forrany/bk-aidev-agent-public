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
  <div class="artifact-card">
    <div class="artifact-card-icon">📄</div>
    <div class="artifact-card-info">
      <span class="artifact-card-name">{{ filename }}</span>
      <span class="artifact-card-size">{{ fileSize }}</span>
    </div>
    <div class="artifact-card-actions">
      <button
        class="artifact-card-btn"
        @click="handlePreview"
      >
        预览
      </button>
      <button
        class="artifact-card-btn"
        @click="handleDownload"
      >
        下载
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
  import { computed } from 'vue';

  import { useArtifactPreviewConsumer } from '../../composables/use-artifact-preview';
  import { useCustomTabConsumer } from '../../composables/use-custom-tab';
  import ArtifactPreview from './artifact-preview.vue';

  const props = defineProps<{
    filename: string;
    htmlContent: string;
  }>();

  const customTab = useCustomTabConsumer();
  const artifactPreview = useArtifactPreviewConsumer();

  const fileSize = computed(() => {
    const bytes = new Blob([props.htmlContent]).size;
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  });

  const handlePreview = () => {
    if (!artifactPreview || !customTab) return;

    const previewId = artifactPreview.createPreview('html', props.htmlContent, {
      filename: props.filename,
    });

    customTab.addCustomTab({
      name: previewId,
      label: props.filename || 'HTML 预览',
      data: {
        component: ArtifactPreview,
        props: { previewId },
      },
    });
  };

  const handleDownload = () => {
    const blob = new Blob([props.htmlContent], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = props.filename || 'untitled.html';
    a.click();
    URL.revokeObjectURL(url);
  };
</script>

<style lang="scss">
  .artifact-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    margin: 8px 0;
    background: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 8px;
    transition: border-color 0.2s;

    &:hover {
      border-color: #3a84ff;
    }
  }

  .artifact-card-icon {
    flex-shrink: 0;
    font-size: 24px;
  }

  .artifact-card-info {
    display: flex;
    flex: 1;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }

  .artifact-card-name {
    overflow: hidden;
    font-size: 13px;
    font-weight: 600;
    color: #313238;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .artifact-card-size {
    font-size: 12px;
    color: #979ba5;
  }

  .artifact-card-actions {
    display: flex;
    flex-shrink: 0;
    gap: 8px;
  }

  .artifact-card-btn {
    padding: 4px 12px;
    font-size: 12px;
    color: #3a84ff;
    cursor: pointer;
    background: #fff;
    border: 1px solid #3a84ff;
    border-radius: 4px;
    transition: all 0.2s;

    &:hover {
      color: #fff;
      background: #3a84ff;
    }
  }
</style>
