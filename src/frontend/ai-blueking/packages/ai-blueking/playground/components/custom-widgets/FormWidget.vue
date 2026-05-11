<template>
  <div class="form-widget">
    <div class="form-title">{{ data.title || '交互表单' }}</div>
    <div class="form-fields">
      <div
        v-for="(field, index) in data.fields"
        :key="index"
        class="form-field"
      >
        <label class="field-label">{{ field.label }}</label>
        <select
          v-if="field.type === 'select'"
          v-model="formValues[field.label]"
          class="field-input"
        >
          <option
            v-for="opt in field.options"
            :key="opt"
            :value="opt"
          >
            {{ opt }}
          </option>
        </select>
        <input
          v-else
          v-model="formValues[field.label]"
          class="field-input"
          :placeholder="field.placeholder || ''"
        />
      </div>
    </div>
    <div class="form-actions">
      <button
        v-for="action in data.actions"
        :key="action"
        class="form-btn"
        :class="{ primary: action === '确认' || action === '提交' }"
        @click="handleAction(action)"
      >
        {{ action }}
      </button>
    </div>
    <div
      v-if="submitted"
      class="form-result"
    >
      已提交: {{ JSON.stringify(formValues, null, 2) }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { reactive, ref } from 'vue';

const props = defineProps<{
  data: {
    title?: string;
    fields?: Array<{
      label: string;
      type?: string;
      options?: string[];
      placeholder?: string;
    }>;
    actions?: string[];
  };
}>();

const formValues = reactive<Record<string, string>>({});
const submitted = ref(false);

// 初始化默认值
(props.data.fields || []).forEach(field => {
  formValues[field.label] = field.options?.[0] || '';
});

const handleAction = (action: string) => {
  if (action === '取消') {
    submitted.value = false;
    return;
  }
  submitted.value = true;
  console.log('[FormWidget] action:', action, 'values:', { ...formValues });
};
</script>

<style scoped>
.form-widget {
  padding: 16px;
  background: #fafbfd;
  border: 1px solid #e1ecff;
  border-radius: 8px;
}

.form-title {
  margin-bottom: 12px;
  font-size: 14px;
  font-weight: 600;
  color: #313238;
}

.form-fields {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 16px;
}

.form-field {
  display: flex;
  gap: 8px;
  align-items: center;
}

.field-label {
  flex-shrink: 0;
  width: 80px;
  font-size: 13px;
  color: #63656e;
}

.field-input {
  flex: 1;
  height: 32px;
  padding: 0 10px;
  font-size: 13px;
  color: #313238;
  background: #fff;
  border: 1px solid #dcdee5;
  border-radius: 4px;
  outline: none;
}

.field-input:focus {
  border-color: #3a84ff;
}

.form-actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

.form-btn {
  height: 32px;
  padding: 0 16px;
  font-size: 13px;
  color: #63656e;
  cursor: pointer;
  background: #fff;
  border: 1px solid #dcdee5;
  border-radius: 4px;
}

.form-btn:hover {
  color: #3a84ff;
  border-color: #3a84ff;
}

.form-btn.primary {
  color: #fff;
  background: #3a84ff;
  border-color: #3a84ff;
}

.form-btn.primary:hover {
  background: #2d6fdf;
}

.form-result {
  padding: 8px 12px;
  margin-top: 12px;
  font-family: monospace;
  font-size: 12px;
  color: #313238;
  white-space: pre-wrap;
  background: #f0f5ff;
  border-radius: 4px;
}
</style>
