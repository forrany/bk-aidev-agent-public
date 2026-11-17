<template>
  <div class="empty-shortcut-view">
    <div class="view-header">
      <h2>空快捷指令划词</h2>
      <p class="view-desc">
        这里传给小鲸的快捷指令固定是空数组。划选下面的文字时，气泡里只应出现「问问小鲸」。
      </p>
    </div>

    <p class="shortcut-state">
      当前 shortcuts：<code>[]</code>
    </p>

    <p class="sample">
      选中这段文字来确认划词效果。快捷指令为空时，选区旁边只出现「问问小鲸」。可以多划几次，包括从句子中间开始选。
    </p>

    <AIBlueking
      :enable-popup="true"
      :error-toast="false"
      :shortcuts="shortcuts"
      url=""
    />
  </div>
</template>

<script setup lang="ts">
  import AIBlueking from '@blueking/ai-blueking';

  import type { IShortcut } from '@blueking/ai-blueking';

  const shortcuts: IShortcut[] = [];
</script>

<style scoped>
  .view-header {
    margin-bottom: 16px;
  }

  .view-header h2 {
    margin: 0 0 8px;
    font-size: 20px;
    font-weight: 600;
    color: #313238;
  }

  .view-desc {
    max-width: 720px;
    margin: 0;
    font-size: 13px;
    line-height: 1.6;
    color: #63656e;
  }

  .shortcut-state {
    margin: 0 0 16px;
    font-size: 13px;
    color: #313238;
  }

  .shortcut-state code {
    padding: 2px 6px;
    font-family: Consolas, 'Liberation Mono', Menlo, monospace;
    color: #3a84ff;
    background: #f0f5ff;
    border-radius: 2px;
  }

  .sample {
    max-width: 720px;
    margin: 0;
    padding: 16px 20px;
    font-size: 16px;
    line-height: 1.8;
    color: #313238;
    background: #fff;
    border: 1px solid #dcdee5;
    border-radius: 4px;
  }
</style>
