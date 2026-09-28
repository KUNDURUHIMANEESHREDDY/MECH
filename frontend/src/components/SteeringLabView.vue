<template>
  <section class="steer-tool" aria-labelledby="steer-title">
    <header class="tool-header">
      <div class="tool-header__copy">
        <p class="tool-eyebrow">Explore / Intervention</p>
        <h1 id="steer-title">Steering Lab</h1>
        <p class="tool-intro">
          Push the residual stream with a live contrast vector: last-token activations for the
          positive prompt minus the negative prompt, added to one block's output. If the next
          token changes, the vector steers the model.
        </p>
      </div>
      <span class="source-pill" :class="`source-pill--${sourceTone}`" role="status">
        <span class="source-dot" aria-hidden="true" />
        {{ sourceLabel }}
      </span>
    </header>

    <form class="steer-form" @submit.prevent="runSteer">
      <label class="steer-field">
        <span>Prompt under test</span>
        <input v-model="prompt" type="text" class="control" placeholder="The capital of France is" />
      </label>
      <label class="steer-field">
        <span>Positive contrast prompt</span>
        <input v-model="posPrompt" type="text" class="control" placeholder="Paris is beautiful." />
      </label>
      <label class="steer-field">
        <span>Negative contrast prompt</span>
        <input v-model="negPrompt" type="text" class="control" placeholder="London is rainy." />
      </label>
      <div class="steer-row">
        <label>Layer
          <select v-model.number="layer" class="control control--inline">
            <option v-for="n in 12" :key="n" :value="n - 1">L{{ n - 1 }}</option>
          </select>
        </label>
        <label>Strength α
          <input v-model.number="alpha" type="number" min="0" max="100" step="1" class="control control--inline control--number" />
        </label>
        <button class="tool-button tool-button--primary" type="submit" :disabled="loading">
          {{ loading ? 'Steering…' : 'Steer' }}
        </button>
      </div>
    </form>

    <div v-if="errorMessage" class="tool-notice tool-notice--error" role="alert">
      <strong>Steering failed</strong>
      <span>{{ errorMessage }}</span>
    </div>

    <section v-if="result" class="steer-result" aria-label="Steering result">
      <div class="verdict" :class="result.flipped ? 'verdict--flipped' : 'verdict--same'" role="status">
        {{ result.flipped ? 'FLIPPED' : 'UNCHANGED' }}
      </div>
      <dl class="readout">
        <div><dt>Clean next token</dt><dd>{{ result.clean_top }}</dd></div>
        <div><dt>Steered next token</dt><dd>{{ result.steered_top }}</dd></div>
        <div><dt>Layer</dt><dd>L{{ result.layer }}</dd></div>
        <div><dt>α</dt><dd>{{ result.alpha }}</dd></div>
        <div><dt>Vector norm</dt><dd>{{ result.vector_norm }}</dd></div>
        <div><dt>Provenance</dt><dd>{{ provenance }}</dd></div>
      </dl>
      <div class="top5-compare">
        <div>
          <span class="subheading">Clean top-5</span>
          <ol>
            <li v-for="(token, index) in result.clean_top5" :key="`c-${index}`"><code>{{ token }}</code></li>
          </ol>
        </div>
        <div>
          <span class="subheading">Steered top-5</span>
          <ol>
            <li v-for="(token, index) in result.steered_top5" :key="`s-${index}`"><code>{{ token }}</code></li>
          </ol>
        </div>
      </div>
    </section>
    <div v-else-if="!loading" class="tool-state" role="status">
      <strong>No steering run yet.</strong>
      <span>Set a contrast pair and press Steer.</span>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { api } from '../services/api';

type JsonRecord = Record<string, unknown>;

interface SteerResult {
  clean_top: string;
  steered_top: string;
  clean_top5: string[];
  steered_top5: string[];
  flipped: boolean;
  layer: number;
  alpha: number;
  vector_norm: number;
}

const prompt = ref('The movie was');
const posPrompt = ref('It was a fantastic wonderful amazing film. The movie was');
const negPrompt = ref('It was a terrible awful horrible film. The movie was');
const layer = ref(8);
const alpha = ref(25);
const loading = ref(false);
const errorMessage = ref('');
const result = ref<SteerResult | null>(null);
const provenance = ref('unavailable');

const sourceTone = computed(() => {
  if (loading.value) return 'loading';
  if (provenance.value === 'live') return 'online';
  return 'offline';
});
const sourceLabel = computed(() => {
  if (loading.value) return 'Steering';
  if (provenance.value === 'live') return 'Live intervention';
  return 'Steering unavailable';
});

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

async function runSteer(): Promise<void> {
  if (loading.value) return;
  loading.value = true;
  errorMessage.value = '';
  result.value = null;
  try {
    const res = await api.gpt2Steer({
      prompt: prompt.value,
      pos_prompt: posPrompt.value,
      neg_prompt: negPrompt.value,
      layer: layer.value,
      alpha: alpha.value,
    });
    if (!isRecord(res) || res.status === 'error' || res.status === 'unavailable') {
      throw new Error(typeof res.error === 'string' ? res.error : 'Steering returned no result.');
    }
    provenance.value = typeof res.provenance === 'string' ? res.provenance : 'unavailable';
    const cleanTop5 = Array.isArray(res.clean_top5) ? res.clean_top5.map(String) : [];
    const steeredTop5 = Array.isArray(res.steered_top5) ? res.steered_top5.map(String) : [];
    result.value = {
      clean_top: String(res.clean_top ?? '—'),
      steered_top: String(res.steered_top ?? '—'),
      clean_top5: cleanTop5,
      steered_top5: steeredTop5,
      flipped: res.flipped === true,
      layer: Number(res.layer ?? 0),
      alpha: Number(res.alpha ?? 0),
      vector_norm: Number(res.vector_norm ?? 0),
    };
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
    provenance.value = 'unavailable';
  } finally {
    loading.value = false;
  }
}
</script>

<style scoped>
.steer-tool {
  display: flex;
  flex-direction: column;
  gap: 14px;
  width: 100%;
  max-width: 880px;
  color: var(--text);
  font-size: 13px;
}

.steer-tool h1,
.steer-tool p {
  margin: 0;
}

.tool-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.tool-eyebrow {
  color: var(--text-muted);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}

.tool-header h1 {
  font-size: 24px;
  line-height: 1.15;
  font-weight: 720;
  letter-spacing: -0.02em;
}

.tool-intro {
  max-width: 720px;
  margin-top: 4px;
  color: var(--text-muted);
  line-height: 1.5;
}

.source-pill {
  display: inline-flex;
  min-height: 26px;
  align-items: center;
  gap: 6px;
  padding: 0 10px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2);
  color: var(--text-muted);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  white-space: nowrap;
}

.source-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
}

.source-pill--online {
  border-color: var(--success);
  color: var(--success);
}

.source-pill--loading {
  color: var(--warning);
}

.source-pill--offline {
  color: var(--danger);
}

.tool-button {
  display: inline-flex;
  min-height: 36px;
  align-items: center;
  padding: 0 14px;
  border: 1px solid var(--border-light);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
}

.tool-button--primary {
  border-color: var(--primary);
  background: var(--primary);
  color: #ffffff;
}

.tool-button--primary:hover:not(:disabled) {
  background: var(--primary-focus);
}

.tool-button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.control {
  min-height: 36px;
  padding: 0 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  font-size: 13px;
}

.control--inline {
  min-height: 30px;
  font-size: 12px;
}

.control--number {
  width: 84px;
}

.steer-form {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 14px;
  border: 1px solid var(--border);
  border-radius: 7px;
  background: var(--surface);
}

.steer-field {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 11px;
  font-weight: 650;
  color: var(--text-dim);
}

.steer-row {
  display: flex;
  align-items: flex-end;
  flex-wrap: wrap;
  gap: 10px;
}

.steer-row label {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: var(--text-muted);
}

.tool-notice {
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-left: 3px solid var(--danger);
  border-radius: 7px;
  background: var(--surface-2);
  font-size: 12px;
}

.steer-result {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px;
  border: 1px solid var(--border);
  border-radius: 7px;
  background: var(--surface);
}

.verdict {
  display: inline-flex;
  align-self: flex-start;
  min-height: 30px;
  align-items: center;
  padding: 0 14px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.08em;
}

.verdict--flipped {
  background: var(--success);
  color: #ffffff;
}

.verdict--same {
  border: 1px dashed var(--text-muted);
  color: var(--text-muted);
}

.readout {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  margin: 0;
}

.readout > div {
  padding: 8px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
}

.readout dt {
  color: var(--text-muted);
  font-size: 9px;
  font-weight: 650;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}

.readout dd {
  margin: 3px 0 0;
  font-family: var(--font-mono);
  font-size: 14px;
  overflow-wrap: anywhere;
}

.top5-compare {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}

.top5-compare .subheading {
  display: block;
  margin-bottom: 4px;
  color: var(--text-muted);
  font-size: 9px;
  font-weight: 650;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}

.top5-compare ol {
  display: flex;
  flex-direction: column;
  gap: 3px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.top5-compare li {
  padding: 4px 8px;
  border: 1px solid var(--border);
  border-radius: 5px;
  background: var(--surface-2);
  font-family: var(--font-mono);
  font-size: 12px;
  overflow-wrap: anywhere;
}

.tool-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 28px 16px;
  color: var(--text-muted);
  text-align: center;
}

@media (max-width: 760px) {
  .tool-header {
    flex-direction: column;
    align-items: stretch;
  }

  .readout {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
