<template>
  <div class="flex flex-col gap-3 font-['system-ui',sans-serif] text-xs text-[var(--ink)]">
    <!-- ── Live prompt bar ────────────── -->
    <div v-if="usesInternalData"
      class="flex gap-2 items-center flex-wrap bg-[var(--canvas)] p-2.5 rounded-lg border border-[var(--border)]">
      <label for="te-prompt" class="text-[var(--ink-muted-48)] font-semibold">Prompt</label>
      <input id="te-prompt" v-model="promptText" type="text" autocomplete="off"
        placeholder="The capital of France is"
        class="flex-1 min-w-[220px] bg-[var(--canvas-parchment)] text-[var(--ink)] border border-[var(--border)] rounded px-2 py-1 text-xs"
        :disabled="runState === 'loading'"
        @keydown.enter="runPrompt" />
      <button @click="runPrompt" :disabled="runState === 'loading' || !promptText.trim()"
        class="px-3 py-1 rounded text-[11px] font-semibold cursor-pointer border border-[var(--border)] bg-[var(--primary)] text-[var(--on-dark)] disabled:opacity-50">
        {{ runState === 'loading' ? 'Running…' : 'Run' }}
      </button>
      <span v-if="runState === 'loading'" role="status" class="text-[var(--ink-muted-48)]">Reading live weights…</span>
      <span v-else-if="runState === 'ready'" class="text-[var(--ink-muted-48)]">Live weights, this session.</span>
      <span v-if="runError" role="alert" class="text-[11px]">{{ runError }}</span>
      <span class="text-[11px] text-[var(--ink-muted-48)]">If the model is not loaded, open Model Explorer and load GPT-2 first.</span>
    </div>
    <!-- ── Controls Bar ────────────── -->
    <div class="flex gap-3 items-center flex-wrap bg-[var(--canvas)] p-2.5 rounded-lg border border-[var(--border)]">
      <label class="flex items-center gap-1">
        <span class="text-[var(--ink-muted-48)] font-semibold">Layer</span>
        <input type="range" :min="0" :max="effNumLayers - 1" v-model.number="selectedLayer"
          @change="onLayerChange"
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
        <span v-if="layerFetching" role="status" class="text-[var(--ink-muted-48)] self-center mr-1">Reading layer…</span>
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
    <div v-if="activeTab === 'attention' && effLayer" class="bg-[var(--canvas)] rounded-lg p-2.5 border border-[var(--border)]">
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
        <template v-if="matrix.length > 0 && effTokens.length > 0">
          <AttentionHeatmap :matrix="matrix" :tokens="effTokens"
            :hoveredToken="hoveredToken" @hoverToken="hoveredToken = $event" />
        </template>
        <div v-else class="text-[var(--ink-muted-48)] text-xs">Press Run above to read live attention.</div>
        <div v-if="hoveredToken !== null && effTokens[hoveredToken]" class="mt-1.5 text-[11px] text-[var(--ink-muted-48)]">
          Hovering: <strong>"{{ effTokens[hoveredToken] }}"</strong> (token {{ hoveredToken }})
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
            :tokens="effTokens"
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
          <template v-if="effLayer?.heads[compareHeads[0]]?.attentionMatrix && effLayer?.heads[compareHeads[1]]?.attentionMatrix">
            <div class="flex gap-2">
              <div class="flex-1">
                <div class="text-[10px] text-[var(--ink-muted-48)] mb-1">Head {{ compareHeads[0] }}</div>
                <AttentionHeatmap :matrix="effLayer.heads[compareHeads[0]].attentionMatrix"
                  :tokens="effTokens" :hoveredToken="null" @hoverToken="() => {}" />
              </div>
              <div class="flex-1">
                <div class="text-[10px] text-[var(--ink-muted-48)] mb-1">Head {{ compareHeads[1] }}</div>
                <AttentionHeatmap :matrix="effLayer.heads[compareHeads[1]].attentionMatrix"
                  :tokens="effTokens" :hoveredToken="null" @hoverToken="() => {}" />
              </div>
            </div>
          </template>
          <div v-else class="text-[var(--ink-muted-48)] text-xs">Press Run above first.</div>
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
      <NeuronUMAPWrapper :points="umapPoints" :tokens="effTokens"
        :selectedId="selectedNeuron != null ? idForHeadNeuron(selectedLayer, selectedHead, selectedNeuron) : null"
        @selectNeuron="handleUmapSelect" :height="460" />
    </div>

    <!-- ── Token Detail ─────────────────── -->
    <div v-if="hoveredToken != null && effTokens[hoveredToken] && neurons.length > 0"
      class="bg-[var(--canvas)] rounded-lg p-3 border border-[var(--border)]">
      <div class="font-semibold text-[var(--ink-muted-48)] uppercase tracking-widest mb-2 text-[11px]">
        Neurons Active for "{{ effTokens[hoveredToken] }}" (token {{ hoveredToken }})
      </div>
      <ActivationHeatmap :activations="neurons.map(n => n.activation)"
        :neuronIndex="selectedNeuron"
        @selectNeuron="(i) => selectedNeuron = i"
        :tokens="effTokens"
        :neuronTokenActivations="neurons.map(n => n.tokenActivations ?? null)" />
    </div>

    <!-- ── Summary Bar ──────────────────── -->
    <div class="flex gap-2 flex-wrap text-[11px] text-[var(--ink-muted-48)]">
      <span>Layers: {{ effNumLayers }}</span>
      <span>Heads/Layer: {{ effNumHeads }}</span>
      <span>Neurons/Layer: {{ neurons.length }}</span>
      <span>Tokens: {{ effTokens.length }}</span>
      <template v-if="selNeuronData">
        <span>Neuron #{{ selNeuronData.index }} activation: {{ selNeuronData.activation.toFixed(4) }}</span>
      </template>
      <template v-if="hoveredToken != null && effTokens[hoveredToken]">
        <span>Token: "{{ effTokens[hoveredToken] }}"</span>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { colors } from '../../design/tokens/colors';
import { api } from '../../services/api';
import AttentionHeatmap from './vue-panels/AttentionHeatmap.vue';
import ActivationHeatmap from './vue-panels/ActivationHeatmap.vue';
import NeuronUMAPWrapper from './neuron-umap/NeuronUMAPWrapper.vue';
import { buildNeuronPoints, idForHeadNeuron, parseNeuronId } from './neuron-umap/data';

/* ── props (the route mounts with none; parent-passed props still win) ── */
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

const usesInternalData = computed(() => (props.layers?.length ?? 0) === 0 && props.numLayers === 0);

/* ── state ──────────────────────────────────────── */
const selectedLayer = ref(0);
const selectedHead = ref(0);
const selectedNeuron = ref<number | null>(null);
const hoveredToken = ref<number | null>(null);
const activeTab = ref<'attention' | 'neurons' | 'spectrum' | 'compare' | 'umap'>('attention');
const compareHeads = ref<number[]>([0, 1]);

const tabs = ['attention', 'neurons', 'spectrum', 'compare', 'umap'] as const;

/* ── live session state (used when routed with no props) ── */
const promptText = ref('The capital of France is');
const runState = ref<'idle' | 'loading' | 'ready' | 'error'>('idle');
const runError = ref('');
const internalTokens = ref<string[]>([]);
const internalLayers = ref<any[]>([]);
const internalNumLayers = ref(0);
const internalNumHeads = ref(0);
const layerFetching = ref(false);

const effTokens = computed(() => usesInternalData.value ? internalTokens.value : props.tokens);
const effLayers = computed(() => usesInternalData.value ? internalLayers.value : props.layers);
const effNumLayers = computed(() => usesInternalData.value ? internalNumLayers.value : props.numLayers);
const effNumHeads = computed(() => usesInternalData.value ? internalNumHeads.value : props.numHeads);

/* ── computed ───────────────────────────────────── */
const effLayer = computed(() => effLayers.value?.[selectedLayer.value]);
const layer = computed(() => effLayer.value);
const head = computed(() => layer.value?.heads[selectedHead.value]);
const neurons = computed(() => head.value?.neurons ?? []);
const matrix = computed(() => head.value?.attentionMatrix ?? []);

const headLabels = computed(() => Array.from({ length: effNumHeads.value }, (_, i) => `H${i}`));

const selNeuronData = computed(() => {
  if (selectedNeuron.value == null) return null;
  return neurons.value[selectedNeuron.value] ?? null;
});

const umapPoints = computed(() => buildNeuronPoints(effLayers.value as any, effTokens.value));

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

/* ── live session fetch (route use; props use skips this) ─── */
function numOr(value: unknown, fallback: number): number {
  const n = typeof value === 'number' && Number.isFinite(value) ? Math.floor(value) : NaN;
  return Number.isInteger(n) && n > 0 ? n : fallback;
}

function okOrThrow(value: any, label: string): any {
  if (!value || typeof value !== 'object') throw new Error(`${label} returned an unreadable response.`);
  const status = typeof value.status === 'string' ? value.status.toLowerCase() : '';
  if (status && status !== 'ok') {
    const msg = typeof value.error === 'string' && value.error
      ? value.error
      : (typeof value.message === 'string' ? value.message : '');
    throw new Error(msg || `${label} returned status "${status}".`);
  }
  return value;
}

function extractTokens(value: any): string[] {
  const pick = (arr: any[]): string[] => arr
    .map((t: any) => (typeof t === 'string' ? t : t?.text ?? t?.token ?? t?.token_str ?? t?.value))
    .filter((t: any): t is string => typeof t === 'string' && t.length > 0);
  if (Array.isArray(value)) return pick(value);
  if (value && typeof value === 'object') {
    for (const key of ['str_tokens', 'tokens', 'token_strings']) {
      if (Array.isArray(value[key])) return pick(value[key]);
    }
  }
  return [];
}

function skeletonLayers(nL: number, nH: number): any[] {
  return Array.from({ length: nL }, (_, li) => ({
    index: li,
    heads: Array.from({ length: nH }, (_, hi) => ({ index: hi, attentionMatrix: [], neurons: [] })),
  }));
}

async function fetchLayerHeads(layerIndex: number): Promise<void> {
  if (!usesInternalData.value || internalNumHeads.value <= 0) return;
  layerFetching.value = true;
  try {
    const settled = await Promise.all(
      Array.from({ length: internalNumHeads.value }, (_, h) =>
        api.gpt2AttentionHead(layerIndex, h).then(
          (v) => ({ h, v }),
          (err) => ({ h, err }),
        )),
    );
    const target = internalLayers.value[layerIndex];
    if (!target) return;
    for (const r of settled) {
      const headSlot = target.heads[r.h];
      if (!headSlot) continue;
      if ('err' in r) { headSlot.attentionMatrix = []; continue; }
      try {
        const v = okOrThrow((r as { v: any }).v, `Attention head L${layerIndex}H${r.h}`);
        headSlot.attentionMatrix = Array.isArray(v.matrix) ? v.matrix : [];
      } catch { headSlot.attentionMatrix = []; }
    }
    try {
      const lv = okOrThrow(await api.gpt2Layer(layerIndex), `Layer ${layerIndex}`);
      const top = Array.isArray(lv.top_active_neurons) ? lv.top_active_neurons : [];
      const neuronsForLayer = top.slice(0, 64).map((n: any) => ({
        index: n.neuron_index ?? n.index ?? 0,
        activation: typeof n.activation === 'number' ? n.activation : 0,
        tokenActivations: null,
      }));
      for (const headSlot of target.heads) headSlot.neurons = neuronsForLayer;
    } catch { /* neurons stay empty; panels say so honestly */ }
  } finally {
    layerFetching.value = false;
  }
}

async function runPrompt(): Promise<void> {
  const text = promptText.value.trim();
  if (!text || runState.value === 'loading') return;
  runState.value = 'loading';
  runError.value = '';
  try {
    const arch = okOrThrow(await api.gpt2Architecture(), 'GPT-2 architecture');
    const nL = numOr(arch.n_layers ?? arch.num_layers, 12);
    const nH = numOr(arch.n_heads ?? arch.num_attention_heads, 12);
    internalNumLayers.value = nL;
    internalNumHeads.value = nH;
    internalLayers.value = skeletonLayers(nL, nH);
    if (selectedLayer.value >= nL) selectedLayer.value = 0;
    if (selectedHead.value >= nH) selectedHead.value = 0;
    if (compareHeads.value.some((h) => h >= nH)) compareHeads.value = [0, Math.min(1, nH - 1)];
    const res = okOrThrow(await api.gpt2RunPrompt(text), 'Prompt run');
    internalTokens.value = extractTokens(res);
    selectedNeuron.value = null;
    hoveredToken.value = null;
    await fetchLayerHeads(selectedLayer.value);
    runState.value = 'ready';
  } catch (e) {
    runState.value = 'error';
    runError.value = e instanceof Error ? e.message : 'Unknown error.';
  }
}

function onLayerChange(): void {
  selectedHead.value = 0;
  selectedNeuron.value = null;
  if (usesInternalData.value && runState.value === 'ready') void fetchLayerHeads(selectedLayer.value);
}

onMounted(() => {
  if (usesInternalData.value) void runPrompt();
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
  const layerChanged = parsed.head !== undefined && parsed.layer !== selectedLayer.value;
  if (parsed.head !== undefined) {
    selectedLayer.value = parsed.layer;
    selectedHead.value = parsed.head;
  }
  selectedNeuron.value = parsed.neuron;
  if (layerChanged && usesInternalData.value && runState.value === 'ready') void fetchLayerHeads(selectedLayer.value);
}
</script>
