<template>
  <div class="flex flex-col h-full bg-[var(--bg)] text-[var(--ink)] overflow-hidden">
    <!-- Header -->
    <div class="flex items-center gap-3 px-4 py-2.5 bg-[var(--surface-pearl)] border-b border-[var(--border)] shrink-0">
      <div>
        <p class="text-[10px] font-bold text-[var(--ink-muted-48)] uppercase tracking-[0.09em]">Explore / Live model</p>
        <h2 class="text-[15px] font-bold text-[var(--ink)]">GPT-2 Live</h2>
      </div>
      <div class="text-[11px] text-[var(--ink-muted-48)]" v-if="arch">
        {{ arch.model_name }} · {{ arch.n_layers }}L · {{ arch.n_heads }}H · d={{ arch.d_model }}
      </div>
      <span v-if="archProvenance !== 'unavailable'" role="status"
        class="ml-auto text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full border"
        :class="archProvenance === 'live'
          ? 'text-[var(--success)] border-[var(--success)]'
          : 'text-[var(--ink-muted-48)] border-[var(--border)]'">
        {{ archProvenance }}
      </span>
    </div>

    <!-- Prompt Bar -->
    <div class="flex items-center gap-2.5 px-4 py-2.5 bg-[var(--surface-pearl)] border-b border-[var(--border)] shrink-0">
      <label for="gpt2-live-prompt" class="text-[10px] text-[var(--ink-muted-48)] uppercase tracking-widest">Prompt</label>
      <input
        id="gpt2-live-prompt"
        v-model="prompt"
        @keydown.enter="runPrompt"
        placeholder="The capital of France is"
        class="flex-1 min-w-0 bg-[var(--bg)] border border-[var(--border)] text-[var(--ink)] rounded-md px-2.5 py-1.5 text-xs"
      />
      <button
        @click="runPrompt"
        :disabled="running || !prompt.trim()"
        class="rounded-lg px-4 py-1.5 text-xs font-semibold text-white cursor-pointer disabled:cursor-default disabled:opacity-65 bg-[var(--primary)] hover:bg-[var(--primary-focus)] disabled:hover:bg-[var(--primary)]"
      >
        {{ running ? 'Running…' : 'Run' }}
      </button>
    </div>

    <!-- Status Message -->
    <div v-if="statusMsg" class="px-4 py-1.5 text-[11px] shrink-0 border-b border-[var(--border)]" role="status"
      :class="statusMsg.ok ? 'text-[var(--primary)] bg-[var(--success)]/[0.09]' : 'text-[var(--danger)] bg-[var(--danger)]/[0.09]'">
      {{ statusMsg.text }}
    </div>

    <!-- Content Area -->
    <div class="flex flex-1 min-h-0">
      <!-- Left: Architecture Panel -->
      <div class="w-[260px] bg-[var(--surface-pearl)] border-r border-[var(--border)] overflow-y-auto p-4 shrink-0">
        <div class="text-[13px] font-bold text-[var(--primary)] mb-3">Architecture</div>

        <template v-if="arch">
          <div class="grid grid-cols-2 gap-1.5 text-[11px] mb-4">
            <span class="text-[var(--ink-muted-48)]">Model</span>
            <span class="text-[var(--body-muted)]">{{ arch.model_name }}</span>
            <span class="text-[var(--ink-muted-48)]">Type</span>
            <span class="text-[var(--body-muted)]">{{ arch.model_type }}</span>
            <span class="text-[var(--ink-muted-48)]">Layers</span>
            <span class="text-[var(--body-muted)]">{{ arch.n_layers }}</span>
            <span class="text-[var(--ink-muted-48)]">Heads</span>
            <span class="text-[var(--body-muted)]">{{ arch.n_heads }}</span>
            <span class="text-[var(--ink-muted-48)]">d_model</span>
            <span class="text-[var(--body-muted)]">{{ arch.d_model }}</span>
            <span class="text-[var(--ink-muted-48)]">d_mlp</span>
            <span class="text-[var(--body-muted)]">{{ arch.d_mlp }}</span>
            <span class="text-[var(--ink-muted-48)]">Vocab</span>
            <span class="text-[var(--body-muted)]">{{ arch.vocab_size }}</span>
          </div>

          <!-- Layer List -->
          <div class="text-[10px] text-[var(--ink-muted-48)] uppercase tracking-widest mb-1.5">Layers</div>
          <div class="flex flex-col gap-0.5" role="listbox" aria-label="Transformer layers">
            <button
              v-for="li in (arch.n_layers || 12)" :key="li - 1"
              @click="selectLayer(li - 1)"
              role="option"
              :aria-selected="selectedLayer === li - 1"
              class="w-full text-left py-[7px] px-3 border-none text-xs cursor-pointer flex items-center gap-1.5"
              :class="selectedLayer === li - 1
                ? 'bg-[var(--accent-soft)] border-l-[3px] border-l-[var(--primary)] text-[var(--primary)]'
                : 'bg-transparent border-l-[3px] border-l-transparent text-[var(--ink)] hover:bg-[var(--accent-soft)]'"
            >
              <span class="text-[var(--ink-muted-48)] text-[10px] font-mono">L{{ li - 1 }}</span>
              <span v-if="layerLoading && selectedLayer === li - 1" class="text-[10px] text-[var(--ink-muted-48)]">reading…</span>
            </button>
          </div>
        </template>
        <div v-else class="text-[11px] text-[var(--ink-muted-48)]">
          {{ archError || 'Reading architecture…' }}
        </div>
      </div>

      <!-- Center: Token Results -->
      <div class="flex-1 overflow-y-auto p-4 bg-[var(--bg)]">
        <!-- Error Display -->
        <div v-if="result?.error" class="bg-[var(--danger)]/[0.09] border border-[var(--danger)]/[0.33] text-[var(--danger)] rounded-lg px-4 py-3 text-xs mb-4" role="alert">
          {{ result.error }}
        </div>

        <!-- Token Strip -->
        <div v-if="result?.str_tokens?.length" class="flex flex-wrap gap-1 mb-4">
          <span
            v-for="(t, i) in result.str_tokens" :key="i"
            class="text-[11px] px-2 py-0.5 rounded font-mono border"
            :class="i === result.str_tokens.length - 1
              ? 'bg-[var(--primary)]/[0.13] border-[var(--primary)] text-[var(--primary)] font-semibold'
              : 'bg-[var(--bg)] border-[var(--border)] text-[var(--ink)]'"
          >
            {{ fmtToken(t) }}
          </span>
        </div>

        <!-- Next Token Prediction -->
        <div v-if="result?.next_token" class="mb-4 p-3 bg-[var(--bg-elev)] rounded-lg border border-[var(--border)]">
          <div class="text-[10px] text-[var(--ink-muted-48)] uppercase tracking-widest mb-1">Next Token · {{ resultProvenance }}</div>
          <div class="text-[18px] font-bold text-[var(--primary)] font-mono">{{ result.next_token }}</div>
        </div>

        <!-- Top-5 Tokens -->
        <div v-if="result?.top5?.length" class="mb-4">
          <div class="text-[10px] text-[var(--ink-muted-48)] uppercase tracking-widest mb-2">Top 5 Predictions</div>
          <div class="grid grid-cols-5 gap-1.5">
            <div
              v-for="(t, i) in result.top5" :key="i"
              class="p-2 rounded bg-[var(--bg)] border border-[var(--border)] text-center"
            >
              <div class="font-mono text-[12px] text-[var(--ink)]">{{ fmtToken(t.token) }}</div>
              <div class="text-[10px] text-[var(--ink-muted-48)]">{{ t.logit?.toFixed(3) }}</div>
            </div>
          </div>
        </div>

        <!-- Empty State -->
        <div v-if="!result && !running" class="flex flex-col items-center justify-center h-full text-[var(--ink-muted-48)]">
          <div class="text-[13px] mb-2">Enter a prompt and click Run</div>
          <div class="text-[11px]">GPT-2 predicts the next token from live weights</div>
        </div>
      </div>

      <!-- Right: Live Layer Detail -->
      <div v-if="selectedLayer !== null" class="w-[300px] bg-[var(--surface-pearl)] border-l border-[var(--border)] overflow-y-auto p-4 shrink-0">
        <div class="text-[13px] font-bold text-[var(--primary)] mb-1">
          Layer {{ selectedLayer }}
        </div>
        <div v-if="layerDetail" class="text-[10px] text-[var(--ink-muted-48)] uppercase tracking-widest mb-3">
          {{ layerDetail.path }} · {{ layerProvenance }} · {{ (layerDetail.n_params || 0).toLocaleString() }} params
        </div>
        <div v-else-if="layerError" class="text-[11px] text-[var(--danger)] mb-3" role="alert">{{ layerError }}</div>
        <div v-else class="text-[11px] text-[var(--ink-muted-48)] mb-3">Select a layer to read its live weights.</div>

        <template v-if="layerDetail">
          <div class="text-[10px] text-[var(--ink-muted-48)] uppercase tracking-widest mb-1.5">Attention Heads · live Q/K/V/O norms</div>
          <div class="grid grid-cols-2 gap-1 mb-4">
            <div
              v-for="h in (layerDetail.attention_heads || [])" :key="h.head_index"
              class="py-1 px-1.5 rounded text-[10px] bg-[var(--bg)] border border-[var(--border)] text-[var(--ink)]"
              :title="`Q ${h.q_weight_l2} · K ${h.k_weight_l2} · V ${h.v_weight_l2} · O ${h.o_weight_l2}`"
            >
              <span class="font-mono font-bold">H{{ h.head_index }}</span>
              <span v-if="h.top_token" class="text-[var(--ink-muted-48)]"> → {{ fmtToken(h.top_token) }}</span>
            </div>
          </div>

          <div class="text-[10px] text-[var(--ink-muted-48)] uppercase tracking-widest mb-1.5">Top Active Neurons</div>
          <div v-if="(layerDetail.top_active_neurons || []).length" class="flex flex-col gap-1 mb-4">
            <div
              v-for="n in layerDetail.top_active_neurons.slice(0, 8)" :key="n.neuron_index"
              class="flex items-baseline gap-2 py-1 px-1.5 rounded text-[10px] bg-[var(--bg)] border border-[var(--border)]"
            >
              <span class="font-mono font-bold text-[var(--ink)]">N{{ n.neuron_index }}</span>
              <span class="font-mono text-[var(--primary)]">{{ n.activation?.toFixed(3) }}</span>
              <span v-if="n.top_token" class="text-[var(--ink-muted-48)] truncate">{{ fmtToken(n.top_token) }}</span>
            </div>
          </div>
          <div v-else class="text-[11px] text-[var(--ink-muted-48)] mb-4">Run a prompt to rank neurons by live activation.</div>

          <div class="text-[10px] text-[var(--ink-muted-48)] uppercase tracking-widest mb-1.5">Info</div>
          <div class="grid grid-cols-2 gap-1.5 text-[11px]">
            <span class="text-[var(--ink-muted-48)]">d_model</span>
            <span class="text-[var(--body-muted)]">{{ arch?.d_model }}</span>
            <span class="text-[var(--ink-muted-48)]">d_mlp</span>
            <span class="text-[var(--body-muted)]">{{ arch?.d_mlp }}</span>
            <span class="text-[var(--ink-muted-48)]">MLP neurons</span>
            <span class="text-[var(--body-muted)]">{{ layerDetail.num_mlp_neurons }}</span>
            <span class="text-[var(--ink-muted-48)]">Activations</span>
            <span class="text-[var(--body-muted)]">{{ layerDetail.has_activations ? 'cached' : 'not populated' }}</span>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { api } from '../services/api';

/* ── helpers ────────────────────────────────────── */
function fmtToken(tok: string) {
  if (!tok) return '';
  return String(tok).replaceAll('Ġ', '␣').replaceAll('Ċ', '⏎').trim();
}

function provenanceOf(value: any): string {
  const raw = value && typeof value.provenance === 'string' ? value.provenance.toLowerCase() : '';
  return raw === 'live' || raw === 'seeded' || raw === 'reference' ? raw : 'unavailable';
}

/* ── state ──────────────────────────────────────── */
const prompt = ref('The capital of France is');
const running = ref(false);
const result = ref<any>(null);
const resultProvenance = ref('unavailable');
const arch = ref<any>(null);
const archProvenance = ref('unavailable');
const archError = ref('');
const selectedLayer = ref<number | null>(null);
const layerDetail = ref<any>(null);
const layerProvenance = ref('unavailable');
const layerLoading = ref(false);
const layerError = ref('');
const statusMsg = ref<{ ok: boolean; text: string } | null>(null);

function note(ok: boolean, text: string) {
  statusMsg.value = { ok, text };
  setTimeout(() => { statusMsg.value = null; }, 4000);
}

/* ── API calls ──────────────────────────────────── */
async function loadArchitecture() {
  archError.value = '';
  try {
    const res: any = await api.gpt2Architecture();
    if (res && res.status === 'error') throw new Error(res.error || 'Architecture unavailable.');
    arch.value = res;
    archProvenance.value = provenanceOf(res);
    if (selectedLayer.value === null) await selectLayer(0);
  } catch (e: any) {
    archError.value = e instanceof Error ? e.message : String(e);
    note(false, `Architecture unavailable: ${archError.value}`);
  }
}

async function selectLayer(layer: number) {
  if (!Number.isInteger(layer) || layer < 0) return;
  selectedLayer.value = layer;
  layerDetail.value = null;
  layerError.value = '';
  layerProvenance.value = 'unavailable';
  layerLoading.value = true;
  try {
    const res: any = await api.gpt2Layer(layer);
    if (res && res.status === 'error') throw new Error(res.error || `Layer ${layer} unavailable.`);
    layerDetail.value = res;
    layerProvenance.value = provenanceOf(res);
  } catch (e: any) {
    layerError.value = e instanceof Error ? e.message : String(e);
  } finally {
    layerLoading.value = false;
  }
}

async function runPrompt() {
  if (!prompt.value.trim() || running.value) return;
  running.value = true;
  statusMsg.value = null;
  try {
    const res: any = await api.gpt2RunPrompt(prompt.value);
    if (res && res.status === 'error') throw new Error(res.error || 'Prompt run failed.');
    result.value = res;
    resultProvenance.value = provenanceOf(res);
    note(true, `Prompt ran — ${res.str_tokens?.length || 0} tokens (${resultProvenance.value})`);
    if (selectedLayer.value !== null) await selectLayer(selectedLayer.value);
  } catch (e: any) {
    result.value = { error: e instanceof Error ? e.message : String(e) };
    resultProvenance.value = 'unavailable';
    note(false, `Error: ${result.value.error}`);
  } finally {
    running.value = false;
  }
}

/* ── load on mount ──────────────────────────────── */
onMounted(() => {
  loadArchitecture();
});
</script>
