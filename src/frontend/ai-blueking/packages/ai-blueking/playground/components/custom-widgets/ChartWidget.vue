<template>
  <div class="chart-widget">
    <div class="chart-title">{{ data.title || '图表' }}</div>
    <v-chart class="chart-instance" :option="chartOption" autoresize />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import VChart from 'vue-echarts';
import { use } from 'echarts/core';
import { BarChart, LineChart, PieChart } from 'echarts/charts';
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  TitleComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

use([
  CanvasRenderer,
  BarChart,
  LineChart,
  PieChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  TitleComponent,
]);

const props = defineProps<{
  data: {
    title?: string;
    chartType?: string;
    data?: {
      labels?: string[];
      values?: number[];
    };
  };
}>();

const colors = ['#3a84ff', '#5ebdff', '#85caff', '#b0d7ff', '#d4e8ff'];

const chartOption = computed(() => {
  const labels = props.data?.data?.labels || [];
  const values = props.data?.data?.values || [];
  const type = props.data?.chartType || 'bar';

  if (type === 'pie') {
    return {
      tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      color: colors,
      series: [
        {
          type: 'pie',
          radius: ['40%', '70%'],
          avoidLabelOverlap: false,
          itemStyle: { borderRadius: 6, borderColor: '#fff', borderWidth: 2 },
          label: { show: true, formatter: '{b}\n{d}%' },
          data: labels.map((label, i) => ({ name: label, value: values[i] || 0 })),
        },
      ],
    };
  }

  const baseOption = {
    tooltip: { trigger: 'axis' },
    color: colors,
    grid: { left: 40, right: 16, top: 16, bottom: 32, containLabel: false },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { color: '#979ba5', fontSize: 11 },
      axisLine: { lineStyle: { color: '#e1ecff' } },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: '#f0f1f5' } },
      axisLabel: { color: '#979ba5', fontSize: 11 },
    },
    series: [
      {
        type,
        data: values,
        smooth: type === 'line',
        barMaxWidth: 40,
        itemStyle: { borderRadius: type === 'bar' ? [4, 4, 0, 0] : undefined },
        areaStyle: type === 'line' ? { color: 'rgba(58,132,255,0.1)' } : undefined,
      },
    ],
  };

  return baseOption;
});
</script>

<style scoped>
.chart-widget {
  padding: 16px;
  background: #fafbfd;
  border: 1px solid #e1ecff;
  border-radius: 8px;
}

.chart-title {
  margin-bottom: 12px;
  font-size: 14px;
  font-weight: 600;
  color: #313238;
}

.chart-instance {
  width: 100%;
  height: 200px;
}
</style>
