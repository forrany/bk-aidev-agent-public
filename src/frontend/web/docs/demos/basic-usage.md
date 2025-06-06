<script setup>
import { onMounted, defineAsyncComponent } from 'vue';

const apiUrl = import.meta.env.BK_API_URL_TMPL || ''


const AIBlueking = defineAsyncComponent({
  loader: () => import('@blueking/ai-blueking'),
});

onMounted(() => {
  // Use dynamic import() which runs only on the client here
  import('@blueking/ai-blueking/dist/vue3/style.css');
});
</script>

# 基础使用示例
:::tip
本页面可以直接体验小鲸的聊天功能，请点击右下角按钮打开小鲸，下面将详细介绍本页面小鲸的代码实现
:::

这个示例展示了 AI 小鲸组件的基本用法，包括如何初始化和调用基本方法。

<ClientOnly>
<AIBlueking :url="apiUrl" />
</ClientOnly>

## 关键代码讲解

### 1. 引入组件
:::code-group

```js [Vue3]
import AIBlueking from '@blueking/ai-blueking';
import '@blueking/ai-blueking/dist/vue3/style.css';
```

```js [Vue2]
import AIBlueking from '@blueking/ai-blueking/vue2';
import '@blueking/ai-blueking/dist/vue2/style.css';
```
:::

### 2. 设置 API 服务地址

```Vue
<template>
<AIBlueking :url="apiUrl" />
</tempalte>

<script setup>
  const apiUrl = 'https://your-api-endpoint.com/assistant/';
</script>
```

API 地址是必须设置的属性，所有与智能体的初始化、对话请求等都通过此连接进行通信。

### 3. 操作方法

```js
// 显示对话窗口
aiBlueking.value.show();

// 隐藏对话窗口
aiBlueking.value.hide();

// 切换显示状态
aiBlueking.value.toggle();

// 发送消息
aiBlueking.value.sendMessage('你好，AI 小鲸！');

// 清空对话
aiBlueking.value.clear();
```

### 4. 监听事件

```vue
<AIBlueking
  @show="handleShow"
  @close="handleClose"
  @message="handleMessage"
  @send="handleSend"
  @error="handleError"
/>
```

## 注意事项

- 确保设置了正确的 API 地址，否则对话功能无法正常使用
- Vue 3 和 Vue 2 版本的使用方式略有不同，请参考对应的示例代码
- 组件默认显示在右下角，可以通过拖拽调整位置 