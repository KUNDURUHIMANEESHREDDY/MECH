<template>
  <div ref="container" :style="{ width: '100%', height: height + 'px' }" />
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, shallowRef } from 'vue';
import type { NeuronUMapProps } from './types';

const props = defineProps<{
  points: NeuronUMapProps['points'];
  selectedId: NeuronUMapProps['selectedId'];
  onSelectNeuron: NeuronUMapProps['onSelectNeuron'];
  height?: number;
  loading?: boolean;
  title?: string;
}>();

const container = ref<HTMLDivElement>();
const root = shallowRef<any>(null);

let NeuronUMAP: any = null;
let createRoot: any = null;

async function loadReact() {
  if (NeuronUMAP) return;
  const [reactMod, reactDOMMod, neuronMod] = await Promise.all([
    import('react'),
    import('react-dom/client'),
    import('./NeuronUMAP'),
  ]);
  createRoot = reactDOMMod.createRoot;
  NeuronUMAP = neuronMod.default;
}

const render = async () => {
  if (!container.value) return;
  await loadReact();
  if (!root.value) {
    root.value = createRoot(container.value);
  }
  const React = (await import('react')).default;
  root.value.render(
    React.createElement(NeuronUMAP, {
      points: props.points,
      selectedId: props.selectedId,
      onSelectNeuron: props.onSelectNeuron,
      height: props.height ?? 480,
      loading: props.loading ?? false,
      title: props.title ?? 'Neuron Map',
    })
  );
};

onMounted(render);
watch(() => [props.points, props.selectedId, props.loading, props.height, props.title], render, { deep: true });
onUnmounted(() => { root.value?.unmount(); root.value = null; });
</script>