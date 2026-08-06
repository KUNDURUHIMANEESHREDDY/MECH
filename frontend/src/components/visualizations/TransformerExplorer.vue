<template>
  <div class="flex flex-col gap-3 font-['system-ui',sans-serif] text-xs text-[var(--ink)]">
    <!-- ── Controls Bar ────────────── -->
    <div class="flex gap-3 items-center flex-wrap bg-[var(--canvas)] p-2.5 rounded-lg border border-[var(--border)]">
      <label class="flex items-center gap-1">
        <span class="text-[var(--ink-muted-48)] font-semibold">Layer</span>
        <input type="range" :min="0" :max="numLayers - 1" v-model.number="selectedLayer"
          @change="selectedHead = 0; selectedNeuron = null"
          class="w-[120px]" />
        <span class="font-mono min-w-[24px] font-bold">{{ selectedLayer }}</span>
      </label>

      <label class="flex items-center gap-1">
        <span class="text-[var(--ink-muted-48)] font-semibold">Head</span>
        <select v-model.number="selectedHead" @change="selectedNeuron = null"
          class="bg-[var(--canvas-parchment)] text-[var(--ink)] border border-[var(--border)] rounded px-1 py-0.5 text-[11px]">
          <option v-for="(l, i) in headLabels" :key="i" :value="i">{{ l }}</option>
        </select>
      </label>

      <label class="flex items-center gap-1">
        <span class="text-[var(--ink-muted-48)] font-semibold">Neuron</span>
        <input type="range" :min="0" :max="Math.max(neurons.length - 1, 0)" v-model.number="selectedNeuron"
          class="w-[120px]" :disabled="neurons.length === 0" />
        <span class="font-mono min-w-[24px] font-bold">{{ selectedNeuron ?? '—' }}</span>
      </label>

      <div class="flex gap-0.5 ml-auto">
        <button v-for="tab in tabs" :key="tab" @click="activeTab = tab"
          class="px-2.5 py-[3px] rounded text-[11px] font-semibold cursor-pointer border border-[var(--border)]"
          :class="activeTab === tab
            ? 'bg-[var(--primary)] text-[var(--on-dark)]'
            : 'bg-[var(--canvas-parchment)] text-[var(--ink)]'">
          {{ tab.charAt(0).toUpperCase() + tab.slice(1) }}
        </button>
      </div>
    </div>

    <!-- ── Multi-Head Overview ──────────── -->
    <div v-if="activeTab === 'attention' && layer" class="bg-[var(--canvas)] rounded-lg p-2.5 border border-[var(--border)]">
      <div class="font-semibold text-[var(--ink-muted-48)] uppercase tracking-widest mb-1.5 text-[11px]">
        Layer {{ selectedLayer }} — All Heads Overview
      </div>
      <div class="flex gap-1 flex-wrap">
        <button v-for="(s, i) in layerHeadSummaries" :key="i"
          @click="selectedHead = i; selectedNeuron = null"
          class="rounded px-2 py-1 cursor-pointer text-[10px] min-w-[36px] text-center border"
          :class="i === selectedHead
            ? 'bg-[var(--primary)] text-[var(--on-dark)] border-[var(--primary)] font-bold'
            : 'bg-[var(--canvas-parchment)] text-[var(--ink)] border-[var(--border)] font-normal'"
          :title="`H${i}: avg=${s.avgActivation.toFixed(3)}, max=${s.maxActivation.toFixed(3)}`">
          H{{ i }}
          <div class="text-[9px] opacity-70">{{ s.avgActivation.toFixed(2) }}</div>
        </button>
      </div>
    </div>

    <!-- ── Main Content (horizontal) ────── -->
    <div class="flex flex-row gap-3 overflow-x-auto items-start">
      <!-- Attention Heatmap -->
      <div v-if="activeTab === 'attention' || activeTab === 'compare'"
        class="bg-[var(--canvas)] rounded-lg p-3 border border-[var(--border)] flex-1 min-w-[340px]">
        <div class="font-semibold text-[var(--ink-muted-48)] uppercase tracking-widest mb-2 text-[11px]">
          Attention Heatmap — L{{ selectedLayer }} H{{ selectedHead }}
        </div>
        <template v-if="matrix.length > 0 && tokens.length > 0">
          <AttentionHeatmap :matrix="matrix" :tokens="tokens"
            :hoveredToken="hoveredToken" @hoverToken="hoveredToken = $event" />
        </template>
        <div v-else class="text-[var(--ink-muted-48)] text-xs">Run a prompt first to populate attention data.</div>
        <div v-if="hoveredToken !== null && tokens[hoveredToken]" class="mt-1.5 text-[11px] text-[var(--ink-muted-48)]">
          Hovering: <strong>"{{ tokens[hoveredToken] }}"</strong> (token {{ hoveredToken }})
        </div>
      </div>

      <!-- Neuron Activation Panel -->
      <div v-if="activeTab === 'neurons' || activeTab === 'spectrum'"
        class="bg-[var(--canvas)] rounded-lg p-3 border border-[var(--border)] flex-1 min-w-[340px]">
        <div class="font-semibold text-[var(--ink-muted-48)] uppercase tracking-widest mb-2 text-[11px]">
          Neuron Activations — Layer {{ selectedLayer }}
        </div>
        <template v-if="neurons.length > 0">
          <ActivationHeatmap :activations="neurons.map(n => n.activation)"
            :neuronIndex="selectedNeuron"
            @selectNeuron="(i) => selectedNeuron = i"
            :tokens="tokens"
            :neuronTokenActivations="neurons.map(n => n.tokenActivations ?? null)" />
        </template>
        <div v-else class="text-[var(--ink-muted-48)] text-xs">No neuron data for this layer.</div>
      </div>

      <!-- Compare Panel -->
      <template v-if="activeTab === 'compare'">
        <div class="bg-[var(--canvas)] rounded-lg p-3 border border-[var(--border)] flex-1 min-w-[340px]">
          <div class="font-semibold text-[var(--ink-muted-48)] uppercase tracking-widest mb-2 text-[11px]">
            Compare Heads — Layer {{ selectedLayer }}
          </div>
          <div class="flex gap-2 mb-2">
            <select v-model.number="compareHeads[0]"
              class="bg-[var(--canvas-parchment)] text-[var(--ink)] border border-[var(--border)] rounded px-1 py-0.5 text-[11px]">
              <option v-for="(l, i) in headLabels" :key="i" :value="i">{{ l }}</option>
            </select>
            <span class="text-[var(--ink-muted-48)] self-center">vs</span>
            <select v-model.number="compareHeads[1]"
              class="bg-[var(--canvas-parchment)] text-[var(--ink)] border border-[var(--border)] rounded px-1 py-0.5 text-[11px]">
              <option v-for="(l, i) in headLabels" :key="i" :value="i">{{ l }}</option>
            </select>
          </div>
          <template v-if="layer?.heads[compareHeads[0]]?.attentionMatrix && layer?.heads[compareHeads[1]]?.attentionMatrix">
            <div class="flex gap-2">
              <div class="flex-1">
                <div class="text-[10px] text-[var(--ink-muted-48)] mb-1">Head {{ compareHeads[0] }}</div>
                <AttentionHeatmap :matrix="layer.heads[compareHeads[0]].attentionMatrix"
                  :tokens="tokens" :hoveredToken="null" @hoverToken="() => {}" />
              </div>
              <div class="flex-1">
                <div class="text-[10px] text-[var(--ink-muted-48)] mb-1">Head {{ compareHeads[1] }}</div>
                <AttentionHeatmap :matrix="layer.heads[compareHeads[1]].attentionMatrix"
                  :tokens="tokens" :hoveredToken="null" @hoverToken="() => {}" />
              </div>
            </div>
          </template>
          <div v-else class="text-[var(--ink-muted-48)] text-xs">Run a prompt first.</div>
        </div>

        <div class="bg-[var(--canvas)] rounded-lg p-3 border border-[var(--border)] flex-1 min-w-[340px]">
          <div class="font-semibold text-[var(--ink-muted-48)] uppercase tracking-widest mb-2 text-[11px]">
            Head Similarity — Layer {{ selectedLayer }}
          </div>
          <div class="flex flex-wrap gap-1">
            <div v-for="(s, i) in layerHeadSummaries" :key="i"
              class="w-9 h-9 rounded flex items-center justify-center text-[9px] font-semibold cursor-pointer border border-[var(--border)]"
              :style="{ background: rgbaPrimary(0.1 + (s.avgActivation / maxAvgActivation) * 0.9),
                         color: (s.avgActivation / maxAvgActivation) > 0.5 ? 'var(--on-dark)' : 'var(--ink)' }"
              @click="selectedHead = i; activeTab = 'attention'"
              :title="`H${i}: avg=${s.avgActivation.toFixed(3)}, max=${s.maxActivation.toFixed(3)}`">
              H{{ i }}
            </div>
          </div>
        </div>
      </template>
    </div>

    <!-- ── Neuron UMAP ─────────────────── -->
    <div v-if="activeTab === 'umap'" class="bg-[var(--canvas)] rounded-lg p-3 border border-[var(--border)] w-full">
      <div class="font-semibold text-[var(--ink-muted-48)] uppercase tracking-widest mb-2 text-[11px]">
        Neuron Map — All Layers &amp; Heads
      </div>
      <NeuronUMAPWrapper :points="umapPoints" :tokens="tokens"
        :selectedId="selectedNeuron != null ? idForHeadNeuron(selectedLayer, selectedHead, selectedNeuron) : null"
        @selectNeuron="handleUmapSelect" :height="460" />
    </div>

    <!-- ── Token Detail ─────────────────── -->
    <div v-if="hoveredToken != null && tokens[hoveredToken] && neurons.length > 0"
      class="bg-[var(--canvas)] rounded-lg p-3 border border-[var(--border)]">
      <div class="font-semibold text-[var(--ink-muted-48)] uppercase tracking-widest mb-2 text-[11px]">
        Neurons Active for "{{ tokens[hoveredToken] }}" (token {{ hoveredToken }})
      </div>
      <ActivationHeatmap :activations="neurons.map(n => n.activation)"
        :neuronIndex="selectedNeuron"
        @selectNeuron="(i) => selectedNeuron = i"
        :tokens="tokens"
        :neuronTokenActivations="neurons.map(n => n.tokenActivations ?? null)" />
    </div>

    <!-- ── Summary Bar ──────────────────── -->
    <div class="flex gap-2 flex-wrap text-[11px] text-[var(--ink-muted-48)]">
      <span>Layers: {{ numLayers }}</span>
      <span>Heads/Layer: {{ numHeads }}</span>
      <span>Neurons/Layer: {{ neurons.length }}</span>
      <span>Tokens: {{ tokens.length }}</span>
      <template v-if="selNeuronData">
        <span>Neuron #{{ selNeuronData.index }} activation: {{ selNeuronData.activation.toFixed(4) }}</span>
      </template>
      <template v-if="hoveredToken != null && tokens[hoveredToken]">
        <span>Token: "{{ tokens[hoveredToken] }}"</span>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { colors } from '../../design/tokens/colors';
import { AttentionHeatmap } from './panels/AttentionHeatmap';
import { ActivationHeatmap } from './panels/ActivationHeatmap';
import NeuronUMAPWrapper from './neuron-umap/NeuronUMAPWrapper.vue';
import { buildNeuronPoints, idForHeadNeuron, parseNeuronId } from './neuron-umap/data';

/* ── props ──────────────────────────────────────── */
const props = withDefaults(defineProps<{
  tokens: string[];
  layers: any[];
  numLayers: number;
  numHeads: number;
}>(), {
  tokens: () => [],
  layers: () => [],
  numLayers: 0,
  numHeads: 0,
});

/* ── state ──────────────────────────────────────── */
const selectedLayer = ref(0);
const selectedHead = ref(0);
const selectedNeuron = ref<number | null>(null);
const hoveredToken = ref<number | null>(null);
const activeTab = ref<'attention' | 'neurons' | 'spectrum' | 'compare' | 'umap'>('attention');
const compareHeads = ref<number[]>([0, 1]);

const tabs = ['attention', 'neurons', 'spectrum', 'compare', 'umap'] as const;

/* ── computed ───────────────────────────────────── */
const layer = computed(() => props.layers?.[selectedLayer.value]);
const head = computed(() => layer.value?.heads[selectedHead.value]);
const neurons = computed(() => head.value?.neurons ?? []);
const matrix = computed(() => head.value?.attentionMatrix ?? []);

const headLabels = computed(() => Array.from({ length: props.numHeads }, (_, i) => `H${i}`));

const selNeuronData = computed(() => {
  if (selectedNeuron.value == null) return null;
  return neurons.value[selectedNeuron.value] ?? null;
});

const umapPoints = computed(() => buildNeuronPoints(props.layers, props.tokens));

const layerHeadSummaries = computed(() => {
  if (!layer.value) return [];
  return layer.value.heads.map((h: any, i: number) => {
    const flat = (h.attentionMatrix as number[] | undefined)?.flat() ?? [];
    return {
      headIndex: i,
      avgActivation: flat.reduce((a: number, b: number) => a + b, 0) / (flat.length || 1),
      maxActivation: Math.max(...(flat.length ? flat : [0])),
    };
  });
});

const maxAvgActivation = computed(() => {
  const vals = layerHeadSummaries.value.map((x: any) => x.avgActivation);
  return Math.max(...vals, 0.01);
});

/* ── helpers ────────────────────────────────────── */
function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rgbaPrimary(a: number): string {
  const [r, g, b] = hexToRgb(colors.primary);
  return `rgba(${r},${g},${b},${a})`;
}

function handleUmapSelect(id: string) {
  const parsed = parseNeuronId(id);
  if (!parsed) { selectedNeuron.value = null; return; }
  if (parsed.head !== undefined) {
    selectedLayer.value = parsed.layer;
    selectedHead.value = parsed.head;
  }
  selectedNeuron.value = parsed.neuron;
}
</script>
