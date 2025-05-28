# 界面定制

AI 小鲸提供了一些选项来定制其外观和行为。

## 自定义标题和欢迎语

您可以通过以下两个属性来自定义AI小鲸的显示标题和初始欢迎语：

- `title` (String): 默认值为 `'AI 小鲸'`。设置头部展示的标题文本。
- `helloText` (String): 默认值为 `'你好，我是小鲸'`。设置初始欢迎页面显示的问候语。

:::code-group
```vue [Vue 3]
<template>
  <AIBlueking
    ref="aiBlueking"
    :url="apiUrl"
    title="企业助手"
    helloText="你好，我是你的企业智能助手"
  />
</template>

<script lang="ts" setup>
import { ref } from 'vue';
import AIBlueking from '@blueking/ai-blueking';
import '@blueking/ai-blueking/dist/vue3/style.css';

const aiBlueking = ref(null);
const apiUrl = 'https://your-api-endpoint.com/assistant/';
</script>
```

```vue [Vue 2]
<template>
  <AIBlueking
    ref="aiBlueking"
    :url="apiUrl"
    title="企业助手"
    helloText="你好，我是你的企业智能助手"
  />
</template>

<script>
import AIBlueking from '@blueking/ai-blueking/vue2';
import '@blueking/ai-blueking/dist/vue2/style.css';

export default {
  components: {
    AIBlueking
  },
  data() {
    return {
      apiUrl: 'https://your-api-endpoint.com/assistant/'
    };
  }
};
</script>
```
:::

## 窗口拖拽与缩放

AI 小鲸的对话窗口默认支持拖拽移动位置和调整大小。用户可以通过拖动窗口标题栏来移动窗口，通过拖动窗口的边缘或右下角来改变窗口尺寸。这是组件的内置功能，无需额外配置。

### 拖拽功能控制与初始位置设置

从 `v0.5.4` 版本开始，您可以通过以下属性来控制组件的拖拽功能以及自定义初始位置和尺寸：

- `draggable` (Boolean): 默认值为 `true`。控制组件是否可拖拽。
- `defaultWidth` (Number): 设置组件初始宽度，像素值（比如`500`）。
- `defaultHeight` (Number): 设置组件初始高度，像素值。
- `defaultTop` (Number): 设置组件初始顶部位置，像素值。
- `defaultLeft` (Number): 设置组件初始左侧位置，像素值。

:::code-group
```vue [Vue 3]
<template>
  <AIBlueking
    ref="aiBlueking"
    :url="apiUrl"
    :draggable="true"
    :default-width="520"
    :default-height="600"
    :default-top="50"
    :default-left="20"
  />
</template>

<script lang="ts" setup>
import { ref } from 'vue';
import AIBlueking from '@blueking/ai-blueking';
import '@blueking/ai-blueking/dist/vue3/style.css';

const aiBlueking = ref(null);
const apiUrl = 'https://your-api-endpoint.com/assistant/';
</script>
```

```vue [Vue 2]
<template>
  <AIBlueking
    ref="aiBlueking"
    :url="apiUrl"
    :draggable="true"
    :default-width="520"
    :default-height="600"
    :default-top="50"
    :default-left="20"
  />
</template>

<script>
import AIBlueking from '@blueking/ai-blueking/vue2';
import '@blueking/ai-blueking/dist/vue2/style.css';

export default {
  components: {
    AIBlueking
  },
  data() {
    return {
      apiUrl: 'https://your-api-endpoint.com/assistant/'
    };
  }
};
</script>
```
:::

## 初始最小化状态

您可以通过 `defaultMinimize` prop 控制 AI 小鲸窗口在首次加载或通过 `handleShow` 方法显示时是否处于最小化状态。

-   `defaultMinimize` (Boolean): 默认值为 `false`。如果设置为 `true`，窗口初始将是最小化状态。

:::code-group
```vue [Vue 3]
<template>
  <AIBlueking
    ref="aiBlueking"
    :url="apiUrl"
    :default-minimize="true"
  />
</template>

<script lang="ts" setup>
import { ref } from 'vue';
import AIBlueking from '@blueking/ai-blueking';
import '@blueking/ai-blueking/dist/vue3/style.css';

const aiBlueking = ref(null);
const apiUrl = 'https://your-api-endpoint.com/assistant/';
</script>
```

```vue [Vue 2]
<template>
  <AIBlueking
    ref="aiBlueking"
    :url="apiUrl"
    :default-minimize="true"
  />
</template>

<script>
import AIBlueking from '@blueking/ai-blueking/vue2';
import '@blueking/ai-blueking/dist/vue2/style.css';

export default {
  components: {
    AIBlueking
  },
  data() {
    return {
      apiUrl: 'https://your-api-endpoint.com/assistant/'
    };
  }
};
</script>
```
:::

## 输入框禁用控制

从 `v1.0.1` 版本开始，您可以通过 `disabledInput` 属性来控制输入框是否处于禁用状态。禁用后，用户将无法在输入框中输入文本或发送消息，适用于只读展示或特定交互场景。

-   `disabledInput` (Boolean): 默认值为 `false`。设置为 `true` 时，输入框将处于禁用状态。

:::code-group
```vue [Vue 3]
<template>
  <AIBlueking
    ref="aiBlueking"
    :url="apiUrl"
    :disabled-input="true"
  />
</template>

<script lang="ts" setup>
import { ref } from 'vue';
import AIBlueking from '@blueking/ai-blueking';
import '@blueking/ai-blueking/dist/vue3/style.css';

const aiBlueking = ref(null);
const apiUrl = 'https://your-api-endpoint.com/assistant/';
</script>
```

```vue [Vue 2]
<template>
  <AIBlueking
    ref="aiBlueking"
    :url="apiUrl"
    :disabled-input="true"
  />
</template>

<script>
import AIBlueking from '@blueking/ai-blueking/vue2';
import '@blueking/ai-blueking/dist/vue2/style.css';

export default {
  components: {
    AIBlueking
  },
  data() {
    return {
      apiUrl: 'https://your-api-endpoint.com/assistant/'
    };
  }
};
</script>
```
:::

### 禁用输入框的应用场景

输入框禁用功能在以下场景中特别有用：

1. **只读展示模式**：
   - 展示预设的对话内容，但不希望用户继续交互
   - 创建演示或教程内容，引导用户了解产品功能

2. **条件限制交互**：
   - 在用户完成特定操作前，暂时禁用输入功能
   - 特定情境下临时限制用户输入

3. **根据权限动态控制**：
   ```vue
   <AIBlueking
     :url="apiUrl"
     :disabled-input="!userHasPermission"
   />
   ```

4. **创建引导式体验**：
   - 结合预设问题，引导用户通过点击预设问题而非直接输入来交互

禁用状态下，输入框会呈现灰色背景，并显示禁用状态的光标样式，为用户提供明确的视觉反馈。
