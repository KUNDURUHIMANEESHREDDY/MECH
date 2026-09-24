<template>
  <div class="explorer-page p-4">
    <h2 class="text-lg font-bold text-[var(--ink)]">Circuit Explorer</h2>
    <p class="text-sm text-[var(--ink-muted-48)] mt-2">Hierarchical exploration of mechanistic circuits.</p>

    <div v-if="loading" class="text-sm text-[var(--ink-muted-48)] mt-4">
      Loading circuits from the backend…
    </div>
    <div v-else-if="error" class="text-sm mt-4" style="color: var(--danger, #ef4444)">
      Cannot reach the circuit registry: {{ error }}. Start the backend first.
    </div>
    <div v-else-if="circuits.length === 0" class="text-sm text-[var(--ink-muted-48)] mt-4">
      No circuits registered.
    </div>

    <div v-else class="mt-4 flex flex-col gap-3">
      <select v-model="selectedId" class="input-text" style="max-width: 420px">
        <option v-for="c in circuits" :key="c.circuit_id" :value="c.circuit_id">
          {{ c.name ?? c.circuit_id }}
        </option>
      </select>

      <div v-if="detail" class="flex flex-col gap-3">
        <div class="sae-subcard">
          <div class="text-xs font-bold">Evidence</div>
          <p class="text-sm mt-1">{{ detail.description ?? '—' }}</p>
        </div>

        <div class="sae-subcard">
          <div class="text-xs font-bold">Faithfulness</div>
          <div class="confidence-bar mt-1">
            <div class="confidence-track">
              <div
                class="confidence-fill"
                :style="{ width: faithfulnessPct + '%', background: 'var(--warning, #eab308)' }"
              />
            </div>
            <span class="confidence-value">{{ faithfulnessLabel }}</span>
          </div>
        </div>

        <div class="sae-subcard">
          <div class="text-xs font-bold mb-2">Attention Heads</div>
          <div class="flex flex-wrap gap-2">
            <span v-if="heads.length === 0" class="text-sm text-[var(--ink-muted-48)]">None listed</span>
            <span v-for="h in heads" :key="h.node_id" class="member-chip" :title="h.description ?? ''">
              {{ h.label }}{{ h.importance_score !== undefined ? ` (${h.importance_score})` : '' }}
            </span>
          </div>
        </div>

        <div class="sae-subcard">
          <div class="text-xs font-bold mb-2">Other Components</div>
          <div class="flex flex-wrap gap-2">
            <span v-if="others.length === 0" class="text-sm text-[var(--ink-muted-48)]">None listed</span>
            <span v-for="n in others" :key="n.node_id" class="member-chip" :title="n.description ?? ''">
              {{ n.label }}
            </span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';

interface CircuitSummary {
  circuit_id: string;
  name?: string;
}

interface CircuitNode {
  node_id: string;
  node_type: string;
  label: string;
  description?: string;
  importance_score?: number;
}

interface CircuitDetail extends CircuitSummary {
  description?: string;
  faithfulness?: number;
  nodes?: CircuitNode[];
}

const circuits = ref<CircuitSummary[]>([]);
const selectedId = ref('');
const detail = ref<CircuitDetail | null>(null);
const loading = ref(true);
const error = ref<string | null>(null);

async function getJSON(path: string): Promise<any> {
  const res = await fetch(`/api${path}`);
  if (!res.ok) throw new Error(`GET ${path}: ${res.status}`);
  return res.json();
}

onMounted(async () => {
  try {
    const data = await getJSON('/circuits');
    const list = Array.isArray(data) ? data : data.circuits ?? [];
    circuits.value = list;
    if (list.length > 0) selectedId.value = list[0].circuit_id;
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
});

watch(selectedId, async id => {
  detail.value = null;
  if (!id) return;
  try {
    detail.value = await getJSON(`/circuits/${encodeURIComponent(id)}`);
  } catch {
    detail.value = null;
  }
});

const heads = computed(() => (detail.value?.nodes ?? []).filter(n => n.node_type === 'attention_head'));
const others = computed(() => (detail.value?.nodes ?? []).filter(n => n.node_type !== 'attention_head'));
const faithfulnessPct = computed(() => Math.round(((detail.value?.faithfulness ?? 0)) * 100));
const faithfulnessLabel = computed(() =>
  detail.value?.faithfulness === undefined || detail.value?.faithfulness === null
    ? '—'
    : `${Math.round((detail.value.faithfulness ?? 0) * 100)}%`,
);
</script>
