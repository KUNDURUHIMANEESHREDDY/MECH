<template>
  <div class="flex flex-col h-full bg-[var(--bg)] text-[var(--ink)] font-['Inter',system-ui,sans-serif] overflow-hidden">
    <!-- Header -->
    <div class="flex items-center gap-3 px-4 py-2.5 bg-[var(--surface-pearl)] border-b border-[var(--border)] shrink-0">
      <h2 class="text-[13px] font-bold text-[var(--purple-border)]">GPT-2 Live</h2>
      <div class="text-[11px] text-[var(--ink-muted-48)]" v-if="arch">
        {{ arch.model_name }} · {{ arch.n_layers }}L · {{ arch.n_heads }}H · d={{ arch.d_model }}
      </div>
    </div>

    <!-- Prompt Bar -->
    <div class="flex items-center gap-2.5 px-4 py-2.5 bg-[var(--surface-pearl)] border-b border-[var(--border)] shrink-0">
      <span class="text-[10px] text-[var(--ink-muted-48)] uppercase tracking-widest">Prompt</span>
      <input
        v-model="prompt"
        @keydown.enter="runPrompt"
        placeholder="Enter prompt..."
        class="flex-1 min-w-0 bg-[var(--bg)] border border-[var(--border)] text-[var(--ink)] rounded-md px-2.5 py-1.5 text-xs"
      />
      <button
        @click="runPrompt"
        :disabled="running || !prompt.trim()"
        class="rounded-lg px-4 py-1.5 text-xs font-semibold text-white cursor-pointer disabled:cursor-default disabled:opacity-65 bg-gradient-to-br from-[var(--purple)] to-[var(--primary)]"
      >
        {{ running ? 'Running…' : 'Run' }}
      </button>
      <button
        @click="loadArchitecture"
        class="bg-transparent border border-[var(--border)] rounded-lg px-3.5 py-1.5 text-xs text-[var(--purple-border)] cursor-pointer"
      >
        Load Architecture
      </button>
    </div>

    <!-- Status Message -->
    <div v-if="statusMsg" class="px-4 py-1.5 text-[11px] shrink-0 border-b border-[var(--border)]"
      :class="statusMsg.ok ? 'text-[var(--primary)] bg-[var(--success)]/[0.09]' : 'text-[var(--danger)] bg-[var(--danger)]/[0.09]'">
      {{ statusMsg.text }}
    </div>

    <!-- Content Area -->
    <div class="flex flex-1 min-h-0">
      <!-- Left: Architecture Panel -->
      <div class="w-[260px] bg-[var(--surface-pearl)] border-r border-[var(--border)] overflow-y-auto p-4 shrink-0">
        <div class="text-[13px] font-bold text-[var(--purple)] mb-3">Architecture</div>

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
          <div class="flex flex-col gap-0.5">
            <button
              v-for="li in (arch.n_layers || 12)" :key="li - 1"
              @click="selectedLayer = li - 1"
              class="w-full text-left py-[7px] px-3 border-none text-xs cursor-pointer flex items-center gap-1.5"
              :class="selectedLayer === li - 1
                ? 'bg-[var(--accent-soft)] border-l-[3px] border-l-[var(--primary)] text-[var(--primary)]'
                : 'bg-transparent border-l-[3px] border-l-transparent text-[var(--ink)] hover:bg-[var(--accent-soft)]'"
            >
              <span class="text-[var(--ink-muted-48)] text-[10px] font-mono">L{{ li - 1 }}</span>
            </button>
          </div>
        </template>
        <div v-else class="text-[11px] text-[var(--ink-muted-48)]">
          Click "Load Architecture" to start.
        </div>
      </div>

      <!-- Center: Token Results -->
      <div class="flex-1 overflow-y-auto p-4 bg-[var(--bg)]">
        <!-- Error Display -->
        <div v-if="result?.error" class="bg-[var(--danger)]/[0.09] border border-[var(--danger)]/[0.33] text-[var(--danger)] rounded-lg px-4 py-3 text-xs mb-4">
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
          <div class="text-[10px] text-[var(--ink-muted-48)] uppercase tracking-widest mb-1">Next Token</div>
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

        <!-- IOI Results -->
        <div v-if="result?.ioi" class="mb-4 p-3 bg-[var(--bg-elev)] rounded-lg border border-[var(--border)]">
          <div class="text-[10px] text-[var(--ink-muted-48)] uppercase tracking-widest mb-2">IOI Analysis</div>
          <div class="grid grid-cols-2 gap-2 text-[11px]">
            <span class="text-[var(--ink-muted-48)]">IO Name</span>
            <span class="text-[var(--ink)]">{{ result.ioi.io_name }}</span>
            <span class="text-[var(--ink-muted-48)]">Subject</span>
            <span class="text-[var(--ink)]">{{ result.ioi.subj_name }}</span>
            <span class="text-[var(--ink-muted-48)]">S-Idiot Head</span>
            <span class="text-[var(--purple)]">{{ result.ioi.s_idiot_head }}</span>
            <span class="text-[var(--ink-muted-48)]">S-Name Mover</span>
            <span class="text-[var(--primary)]">{{ result.ioi.s_name_mover }}</span>
          </div>
        </div>

        <!-- Empty State -->
        <div v-if="!result && !running" class="flex flex-col items-center justify-center h-full text-[var(--ink-muted-48)]">
          <div class="text-[13px] mb-2">Enter a prompt and click Run</div>
          <div class="text-[11px]">GPT-2 will predict the next token</div>
        </div>
      </div>

      <!-- Right: Layer Detail -->
      <div v-if="selectedLayer !== null" class="w-[280px] bg-[var(--surface-pearl)] border-l border-[var(--border)] overflow-y-auto p-4 shrink-0">
        <div class="text-[13px] font-bold text-[var(--purple)] mb-3">
          Layer {{ selectedLayer }}
        </div>

        <div class="text-[10px] text-[var(--ink-muted-48)] uppercase tracking-widest mb-1.5">Attention Heads</div>
        <div class="grid grid-cols-4 gap-1 mb-4">
          <div
            v-for="hi in (arch?.n_heads || 12)" :key="hi - 1"
            class="py-1 px-0.5 rounded text-center text-[10px] bg-[var(--bg)] border border-[var(--border)] text-[var(--ink-muted-48)] cursor-default"
          >
            H{{ hi - 1 }}
          </div>
        </div>

        <div class="text-[10px] text-[var(--ink-muted-48)] uppercase tracking-widest mb-1.5">Info</div>
        <div class="grid grid-cols-2 gap-1.5 text-[11px]">
          <span class="text-[var(--ink-muted-48)]">d_model</span>
          <span class="text-[var(--body-muted)]">{{ arch?.d_model }}</span>
          <span class="text-[var(--ink-muted-48)]">d_mlp</span>
          <span class="text-[var(--body-muted)]">{{ arch?.d_mlp }}</span>
        </div>
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

/* ── state ──────────────────────────────────────── */
const prompt = ref('The capital of France is');
const running = ref(false);
const result = ref<any>(null);
const arch = ref<any>(null);
const selectedLayer = ref<number | null>(null);
const statusMsg = ref<{ ok: boolean; text: string } | null>(null);

/* ── API calls ──────────────────────────────────── */
async function loadArchitecture() {
  try {
    const res = await api.gpt2Architecture();
    arch.value = res;
    statusMsg.value = { ok: true, text: `Loaded ${(res as any).model_name}` };
    setTimeout(() => { statusMsg.value = null; }, 3000);
  } catch (e: any) {
    statusMsg.value = { ok: false, text: `Failed to load: ${e.message}` };
  }
}

async function runPrompt() {
  if (!prompt.value.trim()) return;
  running.value = true;
  statusMsg.value = null;
  try {
    const res = await api.gpt2RunPrompt(prompt.value);
    result.value = res;
    statusMsg.value = { ok: true, text: `Prompt ran — ${(res as any).str_tokens?.length || 0} tokens` };
    setTimeout(() => { statusMsg.value = null; }, 3000);
  } catch (e: any) {
    result.value = { error: e.message };
    statusMsg.value = { ok: false, text: `Error: ${e.message}` };
  } finally {
    running.value = false;
  }
}

/* ── load on mount ──────────────────────────────── */
onMounted(() => {
  loadArchitecture();
});
</script>
