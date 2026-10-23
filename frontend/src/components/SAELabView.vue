<template>
  <section class="sae-surface" data-testid="sae-lab">
    <header class="surface-header">
      <div class="header-copy">
        <p class="surface-kicker">Research / Sparse autoencoders</p>
        <h2 class="surface-title">SAE Lab</h2>
        <p class="surface-description">
          Compatibility first (<code>GET /api/sae/status</code>), then encode one live hidden state
          with the configured checkpoint (<code>POST /api/sae/inspect</code>).
        </p>
      </div>
      <button class="quiet-button" type="button" data-testid="sae-status-refresh" :disabled="statusLoading" @click="loadStatus">
        {{ statusLoading ? 'Probing…' : 'Re-probe status' }}
      </button>
    </header>

    <!-- 1. Compatibility. Nothing about features is claimed before this. -->
    <section class="panel" aria-labelledby="sae-status-title" data-testid="sae-status">
      <header class="panel-heading">
        <div>
          <p class="panel-kicker">Step 1 — capability probe</p>
          <h3 id="sae-status-title">SAE status &amp; compatibility</h3>
          <p class="panel-description">
            Capability flags returned by the backend. A flag is shown as returned; it is never
            inferred here.
          </p>
        </div>
        <span class="source-badge">{{ statusProvenance || 'PROVENANCE NOT RETURNED' }}</span>
      </header>

      <p v-if="statusLoading" class="panel-message" role="status" aria-live="polite">Probing the SAE surface…</p>

      <div v-else-if="statusFailure" class="notice" :class="statusFailure.kind === 'unsupported' ? 'notice-unsupported' : 'notice-error'" role="alert" data-testid="sae-status-failure">
        <strong>{{ statusFailure.kind === 'unsupported' ? 'NOT AVAILABLE' : 'Status probe failed' }}</strong>
        <p>{{ statusFailure.message }}</p>
        <p v-if="statusFailure.kind === 'unsupported'" class="notice-foot">
          This deployment does not serve <code>GET /api/sae/status</code>, so compatibility cannot be
          established and no feature inspection is offered.
        </p>
      </div>

      <template v-else-if="statusRecord">
        <p v-if="!checkpointConfigured" class="notice notice-warn" role="status" data-testid="sae-checkpoint-warning">
          <strong>No SAE checkpoint is configured on the server.</strong>
          The backend returned <code>checkpoint_configured: false</code>. Inspection needs a
          checkpoint: <code>MECH_SAE_SOURCE</code> + <code>MECH_SAE_CHECKPOINT</code> on the server,
          or an explicit source + identifier below. Without one the backend answers NOT AVAILABLE
          and this view shows that answer — it does not show an empty feature set. If the checkpoint
          directory below lists files, choose one to send it explicitly.
        </p>

        <p v-else-if="checkpointIdentifier" class="notice" role="status" data-testid="sae-checkpoint-configured">
          <strong>A checkpoint is configured on the server.</strong>
          An inspection sent without an identifier uses
          <code class="mono">{{ checkpointIdentifier }}</code>.
        </p>

        <table class="status-table" data-testid="sae-compatibility">
          <caption>Compatibility flags exactly as returned by <code>GET /api/sae/status</code>.</caption>
          <thead>
            <tr>
              <th scope="col">Capability</th>
              <th scope="col">State</th>
              <th scope="col">Returned value</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in compatibilityRows" :key="row.key" :data-testid="`sae-status-row-${row.key}`">
              <th scope="row">{{ row.label }}</th>
              <td>
                <span class="flag" :class="`flag-${row.tone}`">{{ row.state }}</span>
              </td>
              <td class="mono">{{ row.value }}</td>
            </tr>
          </tbody>
        </table>

        <dl class="fact-grid" data-testid="sae-checkpoint-facts">
          <div>
            <dt>Checkpoint directory</dt>
            <dd class="mono" data-testid="sae-checkpoint-dir">{{ checkpointDirLabel }}</dd>
          </div>
          <div>
            <dt>Server checkpoint</dt>
            <dd class="mono" data-testid="sae-checkpoint-identifier">{{ checkpointIdentifierLabel }}</dd>
          </div>
          <div>
            <dt>Checkpoints in that directory</dt>
            <dd data-testid="sae-checkpoint-count">{{ checkpointCountLabel }}</dd>
          </div>
        </dl>
      </template>

      <p v-else class="panel-message">The backend has not been probed yet.</p>
    </section>

    <!-- 2. Inspect. -->
    <section class="panel" aria-labelledby="sae-inspect-title">
      <header class="panel-heading">
        <div>
          <p class="panel-kicker">Step 2 — encode one hidden state</p>
          <h3 id="sae-inspect-title">Feature inspection</h3>
          <p class="panel-description">
            The backend takes a live GPT-2 residual or MLP state for this prompt and encodes it with
            the checkpoint's own matrix. Dimensions must match or it refuses.
          </p>
        </div>
      </header>

      <form class="inspect-form" @submit.prevent="inspect">
        <div class="form-row">
          <label class="field-label" for="sae-prompt">Prompt</label>
          <input
            id="sae-prompt"
            v-model="prompt"
            class="text-input"
            type="text"
            autocomplete="off"
            spellcheck="false"
            data-testid="sae-prompt"
            :disabled="isInspecting"
          />
        </div>
        <div class="form-row">
          <label class="field-label" for="sae-layer">Layer (optional)</label>
          <input
            id="sae-layer"
            v-model="layerInput"
            class="text-input short-input"
            type="number"
            min="0"
            step="1"
            placeholder="backend default"
            data-testid="sae-layer"
            :disabled="isInspecting"
          />
        </div>
        <div class="form-row">
          <label class="field-label" for="sae-checkpoint">Checkpoint file</label>
          <select
            id="sae-checkpoint"
            class="text-input"
            ref="checkpointSelect"
            data-testid="sae-checkpoint-select"
            :value="selectedCheckpoint"
            :disabled="!checkpointOptions.length"
            @change="onCheckpointSelected"
          >
            <option value="">— use the server&rsquo;s checkpoint —</option>
            <option v-for="name in checkpointOptions" :key="name" :value="name">{{ name }}</option>
          </select>
        </div>

        <p class="help-copy" data-testid="sae-checkpoint-listing">{{ checkpointListingNote }}</p>

        <div class="form-row">
          <label class="field-label" for="sae-source">Checkpoint source (optional)</label>
          <input
            id="sae-source"
            v-model="sourceInput"
            class="text-input"
            type="text"
            autocomplete="off"
            spellcheck="false"
            placeholder="use server configuration"
            data-testid="sae-source"
            :disabled="isInspecting"
          />
        </div>
        <div class="form-row">
          <label class="field-label" for="sae-identifier">Checkpoint identifier (optional)</label>
          <input
            id="sae-identifier"
            v-model="identifierInput"
            class="text-input"
            type="text"
            autocomplete="off"
            spellcheck="false"
            placeholder="use server configuration"
            data-testid="sae-identifier"
            :disabled="isInspecting"
          />
        </div>
        <div class="form-row">
          <label class="field-label" for="sae-version">Version (optional)</label>
          <input
            id="sae-version"
            v-model="versionInput"
            class="text-input short-input"
            type="text"
            autocomplete="off"
            spellcheck="false"
            placeholder="latest"
            data-testid="sae-version"
            :disabled="isInspecting"
          />
        </div>
        <button class="primary-button" type="submit" data-testid="sae-inspect" :disabled="isInspecting || !prompt.trim() || statusUnavailable">
          {{ isInspecting ? 'Encoding…' : 'Inspect features' }}
        </button>
      </form>

      <p v-if="statusUnavailable" class="help-copy">
        Inspection is disabled because the capability probe could not be read.
      </p>
    </section>

    <p
      v-if="stale"
      class="notice notice-warn"
      role="status"
      data-testid="sae-stale"
    >
      <strong>Stale result.</strong>
      The features below were encoded for “{{ currentRun?.prompt }}” at layer
      {{ currentRun?.layer ?? 'the backend default' }}; the form now differs. Inspect again before
      treating them as the current prompt's features.
    </p>

    <div v-if="isInspecting" class="notice" role="status" aria-live="polite" data-testid="sae-loading">
      <strong>Encoding…</strong>
      <p>The backend is running a live forward pass and applying the checkpoint's encoder.</p>
    </div>

    <div
      v-if="inspectFailure"
      class="notice"
      :class="inspectFailure.kind === 'unsupported' ? 'notice-unsupported' : 'notice-error'"
      role="alert"
      data-testid="sae-inspect-failure"
    >
      <strong>{{ inspectFailure.kind === 'unsupported' ? 'NOT AVAILABLE' : 'Inspection failed' }}</strong>
      <p>{{ inspectFailure.message }}</p>
      <p v-if="inspectFailure.kind === 'unsupported'" class="notice-foot">
        No feature list is drawn for this state. An empty list would read as “no feature fires”,
        which is a measurement the backend did not make.
      </p>
    </div>

    <p v-if="!currentRun && !inspectFailure && !isInspecting" class="notice" data-testid="sae-idle">
      <strong>Idle.</strong> No hidden state has been encoded yet. Status above comes from the
      capability probe; features only appear after a successful inspection.
    </p>

    <template v-if="currentRun">
      <section class="panel" aria-labelledby="sae-run-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Returned run</p>
            <h3 id="sae-run-title">Encoding contract</h3>
          </div>
          <span class="source-badge" data-testid="sae-evidence">{{ asString(currentRun.record.evidence_level) || 'EVIDENCE NOT RETURNED' }}</span>
        </header>
        <dl class="fact-grid" data-testid="sae-run-facts">
          <div><dt>Prompt</dt><dd>{{ asString(currentRun.record.prompt) || currentRun.prompt }}</dd></div>
          <div><dt>Layer</dt><dd>{{ formatInteger(asNumber(currentRun.record.layer)) }}</dd></div>
          <div><dt>Tensor source</dt><dd>{{ asString(currentRun.record.tensor_source) || 'not returned' }}</dd></div>
          <div><dt>d_in / d_sae</dt><dd>{{ formatInteger(asNumber(currentRun.record.d_in)) }} / {{ formatInteger(asNumber(currentRun.record.d_sae)) }}</dd></div>
          <div><dt>Sparsity (k / d_sae)</dt><dd>{{ formatNumber(asNumber(currentRun.record.sparsity), 6) }}</dd></div>
          <div><dt>Weights digest</dt><dd class="mono">{{ asString(currentRun.record.weights_sha256) || 'not returned' }}</dd></div>
        </dl>
        <p class="help-copy">
          Provenance returned by the response: {{ provenanceOf(currentRun.record) || 'not returned' }}.
          <template v-if="fieldProvenanceSummary(currentRun.record)">
            Field provenance: {{ fieldProvenanceSummary(currentRun.record) }}.
          </template>
        </p>
      </section>

      <section class="panel" aria-labelledby="sae-recon-title" data-testid="sae-reconstruction">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Reconstruction</p>
            <h3 id="sae-recon-title">Reconstruction metric</h3>
          </div>
        </header>
        <dl v-if="reconstructionEntries.length" class="fact-grid">
          <div v-for="entry in reconstructionEntries" :key="entry.key">
            <dt>{{ entry.label }}</dt>
            <dd class="mono">{{ formatNumber(entry.value, 6) }} <span class="unit">as returned</span></dd>
          </div>
        </dl>
        <div v-else class="unavailable-block" role="note" data-testid="sae-reconstruction-unavailable">
          <strong>NOT AVAILABLE — reconstruction fidelity</strong>
          <p>
            This response reports <code>reconstruction_measured: {{ asBoolean(currentRun.record.reconstruction_measured) === null ? 'not returned' : String(asBoolean(currentRun.record.reconstruction_measured)) }}</code>
            and carries no reconstruction metric. Reconstruction fidelity was not measured for this
            request, so no fidelity number is shown here.
          </p>
          <p>No number is shown rather than a plausible-looking one.</p>
        </div>
        <p v-if="tensorMismatchNote" class="field-help" role="note" data-testid="sae-tensor-mismatch-note">
          {{ tensorMismatchNote }}
        </p>
      </section>

      <div class="split">
        <section class="panel features-panel" aria-labelledby="sae-features-title">
          <header class="panel-heading">
            <div>
              <p class="panel-kicker">Step 3 — returned features</p>
              <h3 id="sae-features-title">Features for this hidden state</h3>
              <p class="panel-description">
                {{ features.length }} feature(s) returned; select one to record its activation
                examples.
              </p>
            </div>
            <label class="search-label" for="sae-feature-search">
              <span class="field-label">Filter by index</span>
              <input
                id="sae-feature-search"
                v-model="featureFilter"
                class="text-input search-input"
                type="search"
                autocomplete="off"
                placeholder="e.g. 1024"
                data-testid="sae-feature-search"
              />
            </label>
          </header>

          <p v-if="featureMismatch" class="notice notice-warn" role="status" data-testid="sae-features-partial">
            <strong>Partial feature list.</strong>
            {{ featureMismatch }}
          </p>

          <p v-if="features.length === 0" class="panel-message" role="status" data-testid="sae-features-empty">
            The response carried no <code>feature_indices</code>. Zero returned features is not the
            same as zero firing features; nothing is inferred from the absence.
          </p>

          <template v-else>
            <p v-if="filteredFeatures.length === 0" class="panel-message" data-testid="sae-feature-filter-empty">
              No returned feature index matches “{{ featureFilter }}”. The filter only covers the
              features this inspection returned — the backend exposes no arbitrary feature lookup.
            </p>
            <ul v-else class="feature-list" data-testid="sae-features" aria-label="Returned features and their activations">
              <li v-for="feature in filteredFeatures" :key="feature.index">
                <button
                  type="button"
                  class="feature-button"
                  :class="{ 'is-selected': selectedFeature === feature.index }"
                  :aria-current="selectedFeature === feature.index ? 'true' : undefined"
                  :aria-label="`Feature ${feature.index}, activation ${formatNumber(feature.activation, 6)}`"
                  :data-testid="`sae-feature-${feature.index}`"
                  @click="selectedFeature = feature.index"
                >
                  <span class="feature-index mono">#{{ feature.index }}</span>
                  <span class="feature-activation mono">{{ formatNumber(feature.activation, 6) }}</span>
                </button>
              </li>
            </ul>
          </template>
          <p class="help-copy">
            Activation values are the checkpoint's ReLU latent outputs for one hidden state, in raw
            activation units. The filter is a substring match on the returned indices only.
          </p>
        </section>

        <section class="panel" aria-labelledby="sae-examples-title" data-testid="sae-examples">
          <header class="panel-heading">
            <div>
              <p class="panel-kicker">Step 4 — activation examples</p>
              <h3 id="sae-examples-title">
                <template v-if="selectedFeature !== null">Feature #{{ selectedFeature }}</template>
                <template v-else>Select a feature</template>
              </h3>
            </div>
            <span class="source-badge">{{ runHistory.length }} RUN(S) THIS SESSION</span>
          </header>

          <p v-if="selectedFeature === null" class="panel-message">
            Select a returned feature to list the inspections in this session where it fired.
          </p>
          <p v-else-if="exampleRuns.length === 0" class="panel-message" data-testid="sae-examples-empty">
            Feature #{{ selectedFeature }} has no recorded activation in this session's inspections.
          </p>
          <table v-else class="example-table" data-testid="sae-example-table">
            <caption>Inspections in this session where feature #{{ selectedFeature }} was returned.</caption>
            <thead>
              <tr>
                <th scope="col">Prompt</th>
                <th scope="col">Layer</th>
                <th scope="col">Activation</th>
                <th scope="col">Received</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="run in exampleRuns" :key="run.id">
                <td>{{ run.prompt }}</td>
                <td class="mono">{{ run.layer === null ? 'default' : formatInteger(run.layer) }}</td>
                <td class="mono">{{ formatNumber(run.activation, 6) }}</td>
                <td class="mono">{{ formatTimestamp(run.receivedAt) }}</td>
              </tr>
            </tbody>
          </table>
          <p class="help-copy">
            Examples come only from inspections this view performed. The backend exposes no feature
            lookup by index and no cross-corpus activation cache, so none is implied here.
          </p>
        </section>
      </div>
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import { takeResearchSelection } from '../services/researchSelectionContext';
import {
  asArray,
  asBoolean,
  asNumber,
  asString,
  describeError,
  fieldProvenanceSummary,
  formatInteger,
  formatNumber,
  isRecord,
  isUnsupportedError,
  provenanceOf,
  saeInspect,
  saeStatus,
  statusOf,
  type JsonRecord,
} from '../services/interpretabilityClient';

interface Failure {
  kind: 'unsupported' | 'error';
  message: string;
}

interface Feature {
  index: number;
  activation: number | null;
}

interface InspectRun {
  id: number;
  receivedAt: string;
  prompt: string;
  layer: number | null;
  record: JsonRecord;
  features: Feature[];
}

const COMPATIBILITY: Array<{ key: string; label: string; kind: 'flag' | 'text' }> = [
  { key: 'torch_available', label: 'Torch importable', kind: 'flag' },
  { key: 'gpt2_available', label: 'Live GPT-2 loaded', kind: 'flag' },
  { key: 'pipeline_importable', label: 'Training pipeline importable', kind: 'flag' },
  { key: 'checkpoint_configured', label: 'SAE checkpoint configured', kind: 'flag' },
  { key: 'checkpoint_source', label: 'Checkpoint source', kind: 'text' },
  { key: 'training_running', label: 'Training job running', kind: 'flag' },
];

const RECONSTRUCTION_KEYS: Array<{ key: string; label: string }> = [
  { key: 'relative_l2_error', label: 'Relative L2 error (returned)' },
  { key: 'reconstruction_r2', label: 'R² (returned)' },
  { key: 'reconstruction_fvu', label: 'FVU (returned)' },
  { key: 'reconstruction_loss', label: 'Loss (returned)' },
  { key: 'reconstruction_mse', label: 'MSE (returned)' },
];

const statusRecord = ref<JsonRecord | null>(null);
const statusLoading = ref(false);
const statusFailure = ref<Failure | null>(null);

const linkedContext = takeResearchSelection();
const pendingFeature = linkedContext?.feature;
const prompt = ref(linkedContext?.prompt || 'The capital of France is');
const layerInput = ref(linkedContext?.layer == null ? '' : String(linkedContext.layer));
const sourceInput = ref(linkedContext?.source || '');
const identifierInput = ref(linkedContext?.checkpoint || '');
const versionInput = ref(linkedContext?.version || '');

const isInspecting = ref(false);
const inspectFailure = ref<Failure | null>(null);
const currentRun = ref<InspectRun | null>(null);
const runHistory = ref<InspectRun[]>([]);
const featureFilter = ref('');
const selectedFeature = ref<number | null>(null);
let runCounter = 0;
// The server's checkpoint is prefilled into the form once, so a re-probe
// never overwrites what the operator typed.
let prefilledFromStatus = false;

const statusProvenance = computed(() => provenanceOf(statusRecord.value));
const statusUnavailable = computed(() => Boolean(statusFailure.value));

const checkpointConfigured = computed(() => {
  const value = statusRecord.value ? asBoolean(statusRecord.value.checkpoint_configured) : null;
  return value !== false;
});

const compatibilityRows = computed(() =>
  COMPATIBILITY.map((entry) => {
    const raw = statusRecord.value?.[entry.key];
    if (entry.kind === 'text') {
      const text = asString(raw);
      return {
        key: entry.key,
        label: entry.label,
        state: text ? 'returned' : 'not returned',
        value: text || '—',
        tone: text ? 'ok' : 'unknown',
      };
    }
    const flag = asBoolean(raw);
    return {
      key: entry.key,
      label: entry.label,
      state: flag === null ? 'not returned' : flag ? 'available' : 'unavailable',
      value: flag === null ? '—' : String(flag),
      tone: flag === null ? 'unknown' : flag ? 'ok' : 'no',
    };
  }),
);

// -- Checkpoint inventory ----------------------------------------------------
// The status route reports three checkpoint fields. Each is read as
// "reported" or "not reported": a field the backend did not send is
// unknown, which is neither an empty list nor a missing checkpoint.

const NOT_REPORTED = 'not returned by this backend';

function statusHas(key: string): boolean {
  const record = statusRecord.value;
  return Boolean(record) && Object.prototype.hasOwnProperty.call(record, key);
}

const availableCheckpoints = computed<string[] | null>(() => {
  if (!statusHas('available_checkpoints')) return null;
  const raw = statusRecord.value?.available_checkpoints;
  if (!Array.isArray(raw)) return null;
  return raw.filter((item): item is string => typeof item === 'string' && item.trim().length > 0);
});

const checkpointOptions = computed<string[]>(() => availableCheckpoints.value ?? []);

const checkpointDir = computed<string | null>(() => {
  if (!statusHas('checkpoint_dir')) return null;
  const text = asString(statusRecord.value?.checkpoint_dir);
  return text ? text : null;
});

const checkpointIdentifier = computed<string | null>(() => {
  if (!statusHas('checkpoint_identifier')) return null;
  const text = asString(statusRecord.value?.checkpoint_identifier);
  return text ? text : null;
});

const checkpointDirLabel = computed(() => checkpointDir.value ?? NOT_REPORTED);

const checkpointIdentifierLabel = computed(() => {
  if (!statusHas('checkpoint_identifier')) return NOT_REPORTED;
  return checkpointIdentifier.value ?? 'none configured';
});

const checkpointCountLabel = computed(() => {
  const list = availableCheckpoints.value;
  if (list === null) return NOT_REPORTED;
  if (!list.length) return 'none — no .pt files in that directory';
  return `${list.length} checkpoint file(s)`;
});

const checkpointListingNote = computed(() => {
  const list = availableCheckpoints.value;
  if (list === null) {
    return 'This backend does not report which checkpoints exist, so none are listed. Name one explicitly below to point at it.';
  }
  if (!list.length) {
    return 'The checkpoint directory holds no .pt files. A checkpoint can still be named explicitly below.';
  }
  return 'Choosing a file sends it as source “local” with that identifier; the backend resolves the name inside the checkpoint directory.';
});

const selectedCheckpoint = computed(() => {
  const value = identifierInput.value.trim();
  return checkpointOptions.value.includes(value) ? value : '';
});

// A <select> whose options arrive after its value was patched keeps the
// first option selected, so the DOM value is re-applied once the options
// are on screen.
const checkpointSelect = ref<HTMLSelectElement | null>(null);

watch(
  () => [selectedCheckpoint.value, checkpointOptions.value.length],
  () => {
    const element = checkpointSelect.value;
    if (element && element.value !== selectedCheckpoint.value) {
      element.value = selectedCheckpoint.value;
    }
  },
  { flush: 'post' },
);

const features = computed<Feature[]>(() => (currentRun.value ? currentRun.value.features : []));

const filteredFeatures = computed(() => {
  const filter = featureFilter.value.trim();
  if (!filter) return features.value;
  return features.value.filter((feature) => String(feature.index).includes(filter));
});

const featureMismatch = computed(() => {
  if (!currentRun.value) return '';
  const indices = asArray(currentRun.value.record.feature_indices).length;
  const activations = asArray(currentRun.value.record.activations).length;
  if (indices === activations) return '';
  return `${indices} feature index(es) and ${activations} activation(s) were returned; each index without an activation reads “—” rather than a substituted value.`;
});

/**
 * The backend's own caveat about which tensor the number refers to.
 *
 * A checkpoint that records no hook point cannot be refused a layer mismatch,
 * so the reconstruction error is measured against the tensor the route encoded
 * -- which may not be the tensor the dictionary was trained on. The backend
 * reports that fact; the view must not hide it behind a bare number.
 */
const tensorMismatchNote = computed(() => {
  const record = currentRun.value?.record;
  if (!record) return '';
  const note = asString(record.tensor_mismatch_note);
  return note ?? '';
});

const reconstructionEntries = computed(() => {
  const record = currentRun.value?.record;
  if (!record) return [];
  return RECONSTRUCTION_KEYS.map((entry) => ({ ...entry, value: asNumber(record[entry.key]) })).filter(
    (entry): entry is { key: string; label: string; value: number } => entry.value !== null,
  );
});

const stale = computed(() => {
  if (!currentRun.value) return false;
  if (currentRun.value.prompt !== prompt.value.trim()) return true;
  const typedLayer = layerInput.value.trim();
  const runLayer = currentRun.value.layer;
  if (!typedLayer) return false;
  return Number(typedLayer) !== runLayer;
});

const exampleRuns = computed(() => {
  if (selectedFeature.value === null) return [];
  return runHistory.value
    .map((run) => {
      const match = run.features.find((feature) => feature.index === selectedFeature.value);
      return match ? { run, activation: match.activation } : null;
    })
    .filter((entry): entry is { run: InspectRun; activation: number | null } => entry !== null)
    .map(({ run, activation }) => ({
      id: run.id,
      prompt: run.prompt,
      layer: run.layer,
      activation,
      receivedAt: run.receivedAt,
    }));
});

onMounted(() => {
  void loadStatus();
});

async function loadStatus() {
  statusLoading.value = true;
  statusFailure.value = null;
  try {
    const record = await saeStatus();
    if (statusOf(record) !== 'ok') {
      statusRecord.value = null;
      statusFailure.value = {
        kind: statusOf(record) === 'unavailable' ? 'unsupported' : 'error',
        message:
          asString(record.error) ||
          asString(record.reason) ||
          `The backend returned status “${statusOf(record) || 'none'}” without a reason.`,
      };
      return;
    }
    statusRecord.value = record;
    prefillFromStatus();
  } catch (error) {
    statusRecord.value = null;
    statusFailure.value = {
      kind: isUnsupportedError(error) ? 'unsupported' : 'error',
      message: describeError(error),
    };
  } finally {
    statusLoading.value = false;
  }
}

async function inspect() {
  const targetPrompt = prompt.value.trim();
  if (!targetPrompt) return;
  isInspecting.value = true;
  inspectFailure.value = null;
  selectedFeature.value = null;
  featureFilter.value = '';
  try {
    const payload: { prompt: string; layer?: number; source?: string; identifier?: string; version?: string } = {
      prompt: targetPrompt,
    };
    const typedLayer = layerInput.value.trim();
    if (typedLayer) {
      const layer = Number(typedLayer);
      if (!Number.isFinite(layer) || layer < 0) {
        inspectFailure.value = { kind: 'error', message: 'Layer must be a non-negative integer.' };
        return;
      }
      payload.layer = Math.trunc(layer);
    }
    if (sourceInput.value.trim()) payload.source = sourceInput.value.trim();
    if (identifierInput.value.trim()) payload.identifier = identifierInput.value.trim();
    if (versionInput.value.trim()) payload.version = versionInput.value.trim();

    const record = await saeInspect(payload);
    if (statusOf(record) !== 'completed') {
      // The previous run's features belong to another prompt; they must not
      // survive as if they described this one.
      currentRun.value = null;
      inspectFailure.value = {
        kind: statusOf(record) === 'unavailable' ? 'unsupported' : 'error',
        message:
          asString(record.reason) ||
          asString(record.error) ||
          `The backend returned status “${statusOf(record) || 'none'}” without a reason.`,
      };
      return;
    }
    const run: InspectRun = {
      id: (runCounter += 1),
      receivedAt: new Date().toISOString(),
      prompt: asString(record.prompt) || targetPrompt,
      layer: asNumber(record.layer),
      record,
      features: zipFeatures(record),
    };
    currentRun.value = run;
    runHistory.value = [run, ...runHistory.value].slice(0, 20);
    selectedFeature.value = pendingFeature != null && run.features.some((item) => item.index === pendingFeature) ? pendingFeature : null;
    if (pendingFeature != null && !run.features.some((item) => item.index === pendingFeature)) {
      inspectFailure.value = { kind: 'error', message: `Linked feature F${pendingFeature} was not present in this backend response. The result is shown without substituting another feature.` };
    }
  } catch (error) {
    currentRun.value = null;
    inspectFailure.value = {
      kind: isUnsupportedError(error) ? 'unsupported' : 'error',
      message: describeError(error),
    };
  } finally {
    isInspecting.value = false;
  }
}

function zipFeatures(record: JsonRecord): Feature[] {
  const indices = asArray(record.feature_indices);
  const activations = asArray(record.activations);
  const featuresOut: Feature[] = [];
  indices.forEach((rawIndex, position) => {
    const index = asNumber(rawIndex);
    if (index === null) return;
    featuresOut.push({ index, activation: asNumber(activations[position]) });
  });
  return featuresOut;
}

function onCheckpointSelected(event: Event) {
  const value = (event.target as HTMLSelectElement).value;
  if (!value) {
    identifierInput.value = '';
    return;
  }
  identifierInput.value = value;
  sourceInput.value = 'local';
}

/** A bare file name inside the checkpoint directory — not a path. */
function isBareFilename(value: string): boolean {
  const name = value.trim();
  if (!name || name === '.' || name === '..') return false;
  if (name.startsWith('.')) return false;
  if (name.includes('/') || name.includes('\\') || name.includes(':')) return false;
  return true;
}

function prefillFromStatus() {
  if (prefilledFromStatus) return;
  const record = statusRecord.value;
  if (!record) return;
  prefilledFromStatus = true;
  const source = asString(record.checkpoint_source);
  if (source && !sourceInput.value.trim()) sourceInput.value = source;
  const identifier = checkpointIdentifier.value;
  if (identifier && !identifierInput.value.trim() && isBareFilename(identifier)) {
    identifierInput.value = identifier;
  }
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleTimeString();
}
</script>

<style scoped>
.sae-surface {
  min-height: 100%;
  padding: clamp(18px, 3vw, 30px);
  background: var(--surface);
  color: var(--text);
  color-scheme: light;
}

.surface-header,
.panel-heading,
.inspect-form,
.form-row,
.split {
  display: flex;
  gap: 12px;
}

.surface-header,
.panel-heading {
  align-items: flex-start;
  justify-content: space-between;
  flex-wrap: wrap;
}

.surface-header {
  margin-bottom: 18px;
}

.header-copy {
  flex: 1 1 340px;
  min-width: 0;
}

.surface-kicker,
.panel-kicker {
  margin: 0 0 5px;
  color: var(--text-muted);
  font: 700 10px/1.2 var(--font-mono);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.surface-title {
  margin: 0;
  font-size: clamp(22px, 3vw, 30px);
  font-weight: 750;
  letter-spacing: -0.035em;
}

.surface-description,
.panel-description,
.help-copy {
  margin: 7px 0 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.help-copy {
  margin-top: 10px;
  font-size: 11px;
}

.quiet-button,
.primary-button {
  min-height: 32px;
  padding: 0 11px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text-muted);
  cursor: pointer;
  font: 700 11px/1 var(--font);
}

.primary-button {
  align-self: flex-end;
  border-color: var(--primary);
  background: var(--primary);
  color: #ffffff;
}

.quiet-button:hover:not(:disabled) {
  color: var(--text);
  border-color: var(--border-light);
}

.quiet-button:disabled,
.primary-button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.notice {
  margin: 0 0 14px;
  padding: 11px 13px;
  border: 1px solid var(--border);
  border-left: 3px solid var(--border-light);
  border-radius: 6px;
  background: var(--surface-2);
  color: var(--text-dim);
  font-size: 12px;
  line-height: 1.5;
}

.notice strong {
  display: block;
  margin-bottom: 3px;
  color: var(--text);
}

.notice p {
  margin: 0;
}

.notice-warn {
  border-left-color: var(--warning);
}

.notice-error {
  border-left-color: var(--danger);
}

.notice-unsupported {
  border-left-color: var(--text-muted);
  border-style: dashed;
}

.notice-foot {
  margin-top: 6px !important;
  color: var(--text-muted);
  font-size: 11px;
}

.panel {
  margin-bottom: 14px;
  padding: 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.panel-heading {
  margin-bottom: 12px;
}

.panel-heading h3 {
  margin: 0;
  font-size: 15px;
  font-weight: 700;
}

.source-badge {
  padding: 3px 6px;
  border: 1px solid var(--border-light);
  border-radius: 999px;
  color: var(--text-muted);
  font: 700 9px/1 var(--font-mono);
  letter-spacing: 0.06em;
  white-space: nowrap;
}

.status-table,
.example-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
}

.status-table caption,
.example-table caption {
  caption-side: top;
  padding-bottom: 8px;
  color: var(--text-muted);
  font: 400 11px/1.5 var(--font);
  text-align: left;
}

.status-table th,
.status-table td,
.example-table th,
.example-table td {
  padding: 6px 7px;
  border-bottom: 1px solid var(--border);
  text-align: left;
  vertical-align: top;
}

.status-table thead th,
.example-table thead th {
  color: var(--text-muted);
  font: 700 9px/1.4 var(--font-mono);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.flag {
  display: inline-block;
  padding: 2px 6px;
  border: 1px solid var(--border-light);
  border-radius: 999px;
  color: var(--text-muted);
  font: 700 9px/1.4 var(--font-mono);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.flag-ok {
  border-color: var(--success);
  color: var(--success);
}

.flag-no {
  border-color: var(--warning);
  color: var(--warning);
}

.flag-unknown {
  border-style: dashed;
}

.inspect-form {
  align-items: flex-end;
  flex-wrap: wrap;
}

.form-row {
  flex: 1 1 190px;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.field-label {
  color: var(--text-muted);
  font: 700 9px/1.4 var(--font-mono);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.text-input {
  min-height: 32px;
  padding: 0 9px;
  border: 1px solid var(--border-light);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  font: 500 12px/1 var(--font-mono);
}

.short-input {
  max-width: 180px;
}

.search-label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.search-input {
  min-height: 30px;
  font-size: 11px;
}

.fact-grid {
  display: grid;
  gap: 8px 16px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin: 0;
}

.fact-grid dt {
  color: var(--text-muted);
  font: 700 9px/1.4 var(--font-mono);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.fact-grid dd {
  margin: 2px 0 0;
  font-size: 12px;
  line-height: 1.45;
  overflow-wrap: anywhere;
}

.mono {
  font-family: var(--font-mono);
}

.unit {
  color: var(--text-muted);
  font-size: 10px;
}

.unavailable-block {
  padding: 12px;
  border: 1px dashed var(--border-light);
  border-radius: 6px;
  background: var(--surface-2);
  color: var(--text-dim);
  font-size: 12px;
  line-height: 1.55;
}

.unavailable-block strong {
  display: block;
  margin-bottom: 5px;
  color: var(--text);
  font: 700 10px/1.4 var(--font-mono);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.unavailable-block p {
  margin: 0 0 6px;
}

.split {
  align-items: stretch;
  flex-wrap: wrap;
}

.split > .panel {
  flex: 1 1 380px;
  min-width: 0;
}

.feature-list {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin: 0;
  padding: 0;
  list-style: none;
}

.feature-button {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-height: 42px;
  padding: 5px 9px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  cursor: pointer;
  text-align: left;
  font-size: 11px;
}

.feature-button:hover {
  border-color: var(--primary);
}

.feature-button:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 1px;
}

.feature-button.is-selected {
  border-color: var(--primary);
  box-shadow: inset 0 0 0 2px var(--primary);
}

.feature-index {
  font-weight: 700;
}

.feature-activation {
  color: var(--text-muted);
  font-size: 10px;
}

.panel-message {
  margin: 0;
  padding: 12px;
  border: 1px dashed var(--border-light);
  border-radius: 6px;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

@media (max-width: 760px) {
  .split {
    flex-direction: column;
  }

  .split > .panel {
    flex: auto;
  }
}
</style>
