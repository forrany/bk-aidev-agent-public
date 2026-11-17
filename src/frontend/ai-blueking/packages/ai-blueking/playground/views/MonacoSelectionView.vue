<template>
  <div class="monaco-selection-view">
    <div class="view-header">
      <h2>Monaco 0.39.0 划词</h2>
      <p class="view-desc">
        宿主编辑器固定为 monaco-editor 0.39.0。打开划词后，在编辑器里点击、拖选或用方向键移动光标，观察光标是否漂移。
        快捷指令切到空数组后，再划选文本，气泡里只应出现「问问小鲸」。
      </p>
    </div>

    <div class="toolbar">
      <label class="toolbar-item">
        <input
          v-model="enablePopup"
          type="checkbox"
        />
        开启划词
      </label>
      <label class="toolbar-item">
        <input
          v-model="emptyShortcuts"
          type="checkbox"
        />
        快捷指令为 []
      </label>
      <span class="cursor-label">当前光标 {{ cursorLabel }}</span>
    </div>

    <div
      ref="editorRef"
      class="editor"
    />

    <div class="trail">
      <div class="trail-title">光标变化</div>
      <p
        v-if="cursorTrail.length === 0"
        class="trail-empty"
      >
        还没有光标事件。在编辑器中点击或移动光标后，这里会记下行号、列号和来源。
      </p>
      <ol v-else>
        <li
          v-for="(item, index) in cursorTrail"
          :key="`${item}-${index}`"
        >
          {{ item }}
        </li>
      </ol>
    </div>

    <p class="plain-text">
      这是编辑器外面的普通文本。选中这句话可以对比划词气泡：有快捷指令时显示这些指令，指令为空时只显示问问小鲸。
    </p>

    <AIBlueking
      :key="selectionKey"
      :enable-popup="enablePopup"
      :shortcuts="shortcuts"
      :url="requestUrl"
    />
  </div>
</template>

<script setup lang="ts">
  import { computed, onMounted, onUnmounted, ref, useTemplateRef } from 'vue';

  import AIBlueking from '@blueking/ai-blueking';
  import * as monacoNamespace from 'monaco-editor-039/esm/vs/editor/editor.api';
  import EditorWorker from 'monaco-editor-039/esm/vs/editor/editor.worker?worker';

  import 'monaco-editor-039/min/vs/editor/editor.main.css';

  import type { IShortcut } from '@blueking/ai-blueking';

  interface CursorPosition {
    column: number;
    lineNumber: number;
  }

  interface CursorPositionEvent {
    position: CursorPosition;
    source: string;
  }

  interface MonacoEditor {
    dispose: () => void;
    onDidChangeCursorPosition: (listener: (event: CursorPositionEvent) => void) => { dispose: () => void };
  }

  interface MonacoApi {
    editor: {
      create: (
        element: HTMLElement,
        options: {
          automaticLayout: boolean;
          fontSize: number;
          language: string;
          minimap: { enabled: boolean };
          scrollBeyondLastLine: boolean;
          theme: string;
          value: string;
        },
      ) => MonacoEditor;
    };
  }

  const SAMPLE = [
    'function greet(name) {',
    "  const message = 'hello ' + name;",
    '  console.log(message);',
    '  return message;',
    '}',
    '',
    'const users = ["monaco", "selection", "cursor"];',
    'users.forEach(user => greet(user));',
  ].join('\n');

  const DEMO_SHORTCUTS: IShortcut[] = [
    { id: 'explain', name: '解释代码' },
    { id: 'summarize', name: '总结选区' },
  ];

  const monaco = monacoNamespace as unknown as MonacoApi;
  const editorRef = useTemplateRef<HTMLDivElement>('editorRef');
  const enablePopup = ref(true);
  const emptyShortcuts = ref(false);
  const cursorLabel = ref('—');
  const cursorTrail = ref<string[]>([]);
  const apiUrl = import.meta.env.VITE_API_URL || '';

  const shortcuts = computed(() => (emptyShortcuts.value ? [] : DEMO_SHORTCUTS));
  /**
   * 空指令场景不请求 Agent。
   * 否则 props 传入 [] 时会回落到接口指令，页面上看不到「指令为空」。
   */
  const requestUrl = computed(() => (emptyShortcuts.value ? '' : apiUrl));
  const selectionKey = computed(() => (emptyShortcuts.value ? 'empty-shortcuts' : 'demo-shortcuts'));

  const globalScope = globalThis as typeof globalThis & {
    MonacoEnvironment?: {
      getWorker: (workerId: string, label: string) => Worker;
    };
  };

  let editor: MonacoEditor | undefined;
  let cursorListener: { dispose: () => void } | undefined;
  let monacoEnvironmentInstalled = false;
  let previousMonacoEnvironment: typeof globalScope.MonacoEnvironment;

  const recordCursor = (event: CursorPositionEvent) => {
    const { lineNumber, column } = event.position;
    const label = `${lineNumber}:${column} · ${event.source || 'unknown'}`;
    cursorLabel.value = label;
    cursorTrail.value = [label, ...cursorTrail.value].slice(0, 8);
  };

  onMounted(() => {
    const element = editorRef.value;
    if (!element) return;

    previousMonacoEnvironment = globalScope.MonacoEnvironment;
    globalScope.MonacoEnvironment = {
      getWorker: () => new EditorWorker(),
    };
    monacoEnvironmentInstalled = true;

    editor = monaco.editor.create(element, {
      value: SAMPLE,
      language: 'plaintext',
      theme: 'vs',
      automaticLayout: true,
      fontSize: 14,
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
    });
    cursorListener = editor.onDidChangeCursorPosition(recordCursor);
  });

  onUnmounted(() => {
    cursorListener?.dispose();
    editor?.dispose();
    if (monacoEnvironmentInstalled) {
      globalScope.MonacoEnvironment = previousMonacoEnvironment;
    }
  });
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
    max-width: 760px;
    margin: 0;
    font-size: 13px;
    line-height: 1.6;
    color: #63656e;
  }

  .toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: center;
    margin-bottom: 12px;
  }

  .toolbar-item {
    display: flex;
    gap: 6px;
    align-items: center;
    font-size: 13px;
    color: #313238;
    cursor: pointer;
  }

  .cursor-label {
    font-family: Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: 12px;
    color: #3a84ff;
  }

  .editor {
    width: 100%;
    max-width: 880px;
    height: 320px;
    overflow: hidden;
    border: 1px solid #dcdee5;
    border-radius: 4px;
  }

  .trail {
    max-width: 880px;
    margin-top: 12px;
    padding: 12px 16px;
    background: #fff;
    border: 1px solid #dcdee5;
    border-radius: 4px;
  }

  .trail-title {
    margin-bottom: 8px;
    font-size: 13px;
    font-weight: 600;
    color: #313238;
  }

  .trail-empty {
    margin: 0;
    font-size: 12px;
    color: #979ba5;
  }

  .trail ol {
    margin: 0;
    padding-left: 18px;
    font-family: Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: 12px;
    line-height: 1.7;
    color: #63656e;
  }

  .plain-text {
    max-width: 880px;
    margin-top: 16px;
    font-size: 14px;
    line-height: 1.7;
    color: #313238;
  }
</style>
