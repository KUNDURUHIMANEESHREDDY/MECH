<template>
  <section
    class="model-explorer"
    aria-labelledby="model-explorer-title"
    :aria-busy="isBusy"
  >
    <div class="model-explorer__frame">
      <header class="explorer-header">
        <div class="explorer-header__identity">
          <span class="explorer-kicker">Interpretability / runtime</span>
          <h1 id="model-explorer-title">Model Explorer</h1>
          <p>
            Inspect the model registry, run a prompt, and follow the returned
            tokens into GPT-2 layers, heads, and neurons.
          </p>
        </div>
        <div class="explorer-header__actions">
          <div
            class="runtime-state"
            :class="`runtime-state--${runtimeState}`"
            role="status"
            aria-live="polite"
          >
            <span class="runtime-state__dot" aria-hidden="true" />
            <span>{{ runtimeLabel }}</span>
          </div>
          <a class="society-link" href="#society" data-testid="model-explorer-society">
            <span>Open Society</span>
            <span aria-hidden="true">↗</span>
            <span class="sr-only"> (opens the Society tool)</span>
          </a>
        </div>
      </header>

      <div
        v-if="offlineNotice"
        class="state-banner state-banner--offline"
        role="alert"
      >
        <strong>Backend unavailable</strong>
        <span>{{ offlineNotice }}</span>
      </div>
      <div
        v-else-if="catalogError"
        class="state-banner state-banner--error"
        role="alert"
      >
        <strong>Model registry could not be read.</strong>
        <span>{{ catalogError }}</span>
      </div>

      <section class="panel catalog-panel" aria-labelledby="model-catalog-title">
        <div class="panel-heading">
          <div>
            <span class="panel-kicker">Runtime registry</span>
            <h2 id="model-catalog-title">Choose a model</h2>
          </div>
          <div class="panel-heading__actions">
            <span
              class="provenance-badge"
              :class="`provenance-badge--${catalogProvenance}`"
              :title="provenanceTitle(catalogProvenance)"
            >
              <span class="provenance-badge__dot" aria-hidden="true" />
              {{ catalogProvenance }}
            </span>
            <button
              class="button button--quiet"
              type="button"
              :disabled="catalogLoading || isOffline"
              @click="loadCatalog"
            >
              {{ catalogLoading ? 'Refreshing…' : 'Refresh catalog' }}
            </button>
          </div>
        </div>

        <div v-if="catalogLoading" class="loading-state" role="status">
          <span class="spinner" aria-hidden="true" />
          <span>Reading the model registry from the MECH runtime…</span>
        </div>
        <div v-else-if="models.length" class="catalog-content">
          <div class="catalog-meta">
            <span>{{ models.length }} {{ models.length === 1 ? 'model' : 'models' }} returned by <code>/api/models</code></span>
            <span v-if="catalogProvenance === 'reference'">Catalog metadata only; loading a model is required for model data.</span>
            <span v-else-if="catalogProvenance === 'seeded'">The backend marked this response as seeded.</span>
            <span v-if="formatFieldProvenance(catalogFieldProvenance) !== 'Unavailable'">Field provenance: {{ formatFieldProvenance(catalogFieldProvenance) }}</span>
          </div>

          <ul class="model-list" aria-label="Available models">
            <li v-for="model in models" :key="model">
              <button
                class="model-option"
                type="button"
                :class="{ 'model-option--selected': selectedModel === model }"
                :aria-pressed="selectedModel === model"
                @click="selectModel(model)"
              >
                <span class="model-option__name">{{ model }}</span>
                <span class="model-option__state">
                  {{ sameModel(loadedModel, model) ? 'loaded' : selectedModel === model ? 'selected' : 'available' }}
                </span>
              </button>
            </li>
          </ul>

          <div class="model-load-controls">
            <div class="field-group field-group--model">
              <label for="model-explorer-model">Active model</label>
              <select
                id="model-explorer-model"
                v-model="selectedModel"
                class="control"
                :disabled="isOffline"
                @change="onModelSelectionChange"
              >
                <option value="" disabled>Select a model</option>
                <option v-for="model in models" :key="`select-${model}`" :value="model">
                  {{ model }}
                </option>
              </select>
            </div>
            <button
              class="button button--primary"
              type="button"
              :disabled="!canLoadModel"
              @click="loadSelectedModel"
            >
              <span v-if="modelLoadState === 'loading'" class="spinner spinner--small" aria-hidden="true" />
              {{ modelLoadState === 'loading' ? 'Loading…' : sameModel(loadedModel, selectedModel) ? 'Reload model' : 'Load model' }}
            </button>
            <div v-if="loadedModel" class="loaded-model" role="status" aria-live="polite">
              <span>Loaded</span>
              <strong>{{ loadedModel }}</strong>
              <span
                class="provenance-badge provenance-badge--compact"
                :class="`provenance-badge--${modelProvenance}`"
                :title="provenanceTitle(modelProvenance)"
              >
                <span class="provenance-badge__dot" aria-hidden="true" />
                {{ modelProvenance }}
              </span>
            </div>
            <div v-else class="loaded-model loaded-model--empty">No model is loaded in this view.</div>
            <p v-if="formatFieldProvenance(modelFieldProvenance) !== 'Unavailable'" class="field-help">Field provenance: {{ formatFieldProvenance(modelFieldProvenance) }}</p>
          </div>
          <p v-if="modelLoadError" class="inline-error" role="alert">{{ modelLoadError }}</p>
          <p v-else-if="modelLoadNote" class="field-help">{{ modelLoadNote }}</p>
        </div>
        <div v-else class="empty-state empty-state--compact">
          <strong>No models returned.</strong>
          <span>The runtime registry is reachable but did not provide a usable model list.</span>
        </div>
      </section>

      <section class="panel prompt-panel" aria-labelledby="prompt-title">
        <div class="panel-heading">
          <div>
            <span class="panel-kicker">Inference console</span>
            <h2 id="prompt-title">Prompt workbench</h2>
          </div>
          <span
            class="provenance-badge"
            :class="`provenance-badge--${promptProvenance}`"
            :title="provenanceTitle(promptProvenance)"
          >
            <span class="provenance-badge__dot" aria-hidden="true" />
            {{ promptProvenance }}
          </span>
        </div>

        <form class="prompt-form" novalidate @submit.prevent="runPrompt()">
          <div class="prompt-label-row">
            <label for="model-explorer-prompt">Prompt</label>
            <span class="keyboard-hint">Ctrl / ⌘ + Enter to run</span>
          </div>
          <textarea
            id="model-explorer-prompt"
            v-model="prompt"
            class="control control--textarea"
            rows="3"
            :aria-invalid="Boolean(promptError)"
            :aria-describedby="promptError ? 'model-explorer-prompt-help model-explorer-prompt-error' : 'model-explorer-prompt-help'"
            @keydown="onPromptKeydown"
          />
          <div class="prompt-controls">
            <div class="field-group field-group--endpoint">
              <label for="model-explorer-endpoint">Run endpoint</label>
              <select id="model-explorer-endpoint" v-model="runMode" class="control">
                <option value="gpt2">GPT-2 run prompt</option>
                <option value="infer">Generic infer</option>
              </select>
            </div>
            <button class="button button--primary prompt-run" type="submit" :disabled="!canRunPrompt">
              <span v-if="promptState === 'loading'" class="spinner spinner--small" aria-hidden="true" />
              {{ promptState === 'loading' ? 'Running…' : 'Run prompt' }}
            </button>
            <button
              class="button button--secondary"
              type="button"
              :disabled="freshPromptState === 'loading'"
              @click="fetchFreshPrompt(true)"
            >
              <span v-if="freshPromptState === 'loading'" class="spinner spinner--small" aria-hidden="true" />
              {{ freshPromptState === 'loading' ? 'Creating…' : 'New prompt' }}
            </button>
            <button
              class="button button--secondary"
              type="button"
              :disabled="!canRunPrompt"
              @click="runPrompt('infer')"
            >
              Run generic infer
            </button>
            <button
              class="button button--secondary"
              type="button"
              :disabled="!canRunPrompt"
              @click="runPrompt('gpt2')"
            >
              Run GPT-2
            </button>
          </div>
          <p id="model-explorer-prompt-help" class="field-help">
            Results are shown only when the backend returns them. A seeded response is labeled and is not presented as a live measurement.
            After each run, the predicted next token is appended to the prompt, so repeated runs generate text token by token.
          </p>
          <label class="auto-prompt-toggle">
            <input v-model="autoPrompt" type="checkbox" />
            <span>New model-created prompt every minute</span>
          </label>
          <p v-if="freshPromptNote" class="provenance-note">{{ freshPromptNote }}</p>
          <p v-if="promptResult" class="provenance-note">{{ promptProvenanceNote }}</p>
          <p v-if="promptResult && formatFieldProvenance(promptFieldProvenance) !== 'Unavailable'" class="provenance-note">Field provenance: {{ formatFieldProvenance(promptFieldProvenance) }}</p>
          <p v-if="promptError" id="model-explorer-prompt-error" class="inline-error" role="alert">
            {{ promptError }}
          </p>
        </form>

        <div class="prompt-result" :aria-busy="promptState === 'loading'">
          <div v-if="promptState === 'loading'" class="loading-state loading-state--result" role="status">
            <span class="spinner" aria-hidden="true" />
            <span>Running the selected endpoint…</span>
          </div>
          <div v-else-if="promptResult" class="result-content">
            <div class="result-meta">
              <span>
                {{ promptSource === 'infer' ? 'Generic inference' : 'GPT-2 prompt run' }}
                <template v-if="resultModelName"> · {{ resultModelName }}</template>
              </span>
              <span v-if="resultTokenCount">{{ resultTokenCount }} returned tokens</span>
            </div>

            <div v-if="resultTokens.length" class="token-output">
              <div class="subheading">Returned token sequence</div>
              <ol class="token-strip" aria-label="Returned tokens">
                <li v-for="(token, index) in resultTokens" :key="`${index}-${token}`" class="token-chip">
                  <span class="token-chip__index">{{ index }}</span>
                  <span>{{ formatToken(token) }}</span>
                </li>
              </ol>
            </div>
            <div v-else class="empty-state empty-state--inline">
              <strong>No token sequence returned.</strong>
              <span>The endpoint completed without a readable token list.</span>
            </div>

            <div v-if="nextToken" class="next-token-card">
              <div>
                <span class="subheading">Next token</span>
                <strong>{{ formatToken(nextToken) }}</strong>
              </div>
              <span class="next-token-card__source">{{ nextTokenSource }}</span>
            </div>

            <div v-if="predictions.length" class="prediction-section">
              <div class="subheading">Top returned predictions</div>
              <ol class="prediction-list" aria-label="Top returned predictions">
                <li v-for="(prediction, index) in predictions" :key="`${index}-${prediction.token}`" class="prediction-row">
                  <span class="prediction-row__rank">{{ String(index + 1).padStart(2, '0') }}</span>
                  <code>{{ formatToken(prediction.token) }}</code>
                  <span v-if="prediction.logit !== undefined" class="prediction-row__value">logit {{ formatNumber(prediction.logit) }}</span>
                  <span v-if="prediction.probability !== undefined" class="prediction-row__value">p {{ formatProbability(prediction.probability) }}</span>
                </li>
              </ol>
            </div>
          </div>
          <div v-else class="empty-state empty-state--result">
            <span class="empty-state__mark" aria-hidden="true">→</span>
            <strong>No prompt result yet.</strong>
            <span>Enter a prompt, choose an endpoint, and run it to populate the token and inspector views.</span>
          </div>
        </div>
      </section>

      <div class="workspace-grid">
        <aside class="panel architecture-panel" aria-labelledby="architecture-title">
          <div class="panel-heading panel-heading--compact">
            <div>
              <span class="panel-kicker">Model topology</span>
              <h2 id="architecture-title">Architecture</h2>
            </div>
            <span
              v-if="architecture"
              class="provenance-badge provenance-badge--compact"
              :class="`provenance-badge--${architectureProvenance}`"
              :title="provenanceTitle(architectureProvenance)"
            >
              <span class="provenance-badge__dot" aria-hidden="true" />
              {{ architectureProvenance }}
            </span>
          </div>

          <div v-if="architectureLoading" class="loading-state loading-state--side" role="status">
            <span class="spinner" aria-hidden="true" />
            <span>Loading architecture metadata…</span>
          </div>
          <div v-else-if="architecture" class="architecture-content">
            <dl v-if="architectureRows.length" class="metadata-list">
              <div v-for="row in architectureRows" :key="row.label" class="metadata-row">
                <dt>{{ row.label }}</dt>
                <dd>{{ row.value }}</dd>
              </div>
            </dl>

            <div class="selector-block">
              <label for="model-explorer-layer">Layer</label>
              <select
                id="model-explorer-layer"
                class="control"
                :value="selectedLayer ?? ''"
                :disabled="!layerOptions.length"
                @change="onLayerChange"
              >
                <option value="" disabled>Select a layer</option>
                <option v-for="layer in layerOptions" :key="`layer-${layer}`" :value="layer">
                  Layer {{ layer }}
                </option>
              </select>
            </div>

            <div class="selector-block">
              <label for="model-explorer-head">Attention head</label>
              <select
                id="model-explorer-head"
                class="control"
                :value="selectedHead ?? ''"
                :disabled="!headOptions.length"
                @change="onHeadChange"
              >
                <option value="" disabled>Select a head</option>
                <option v-for="head in headOptions" :key="`head-${selectedLayer}-${head}`" :value="head">
                  Head {{ head }}
                </option>
              </select>
            </div>

            <div class="selector-block">
              <div class="selector-label-row">
                <label for="model-explorer-neuron">Neuron index</label>
                <span v-if="maxNeuronInput !== undefined" class="field-help field-help--inline">0–{{ maxNeuronInput }}</span>
              </div>
              <div class="neuron-input-row">
                <input
                  id="model-explorer-neuron"
                  v-model="neuronIndexInput"
                  class="control"
                  type="number"
                  min="0"
                  :max="maxNeuronInput"
                  :disabled="!isGpt2Model"
                  placeholder="e.g. 42"
                  @change="onNeuronInputChange"
                />
                <button class="button button--secondary" type="button" :disabled="!canInspectNeuron" @click="inspectNeuronInput">
                  Inspect
                </button>
              </div>
            </div>

            <div v-if="neuronOptions.length" class="neuron-shortcuts">
              <span class="field-help">Returned active neurons</span>
              <div class="neuron-chip-list">
                <button
                  v-for="neuron in neuronOptions"
                  :key="`neuron-option-${neuron.index}`"
                  class="neuron-chip"
                  type="button"
                  :class="{ 'neuron-chip--selected': selectedNeuron === neuron.index }"
                  @click="selectNeuron(neuron.index)"
                >
                  N{{ neuron.index }}
                </button>
              </div>
            </div>
          </div>
          <div v-else class="empty-state empty-state--side">
            <strong>Architecture unavailable.</strong>
            <span>{{ architectureHint }}</span>
          </div>
          <p v-if="architectureError" class="inline-error" role="alert">{{ architectureError }}</p>
        </aside>

        <div class="inspector-column">
          <section class="panel attention-panel" aria-labelledby="attention-title">
            <div class="panel-heading panel-heading--compact">
              <div>
                <span class="panel-kicker">Attention readout</span>
                <h2 id="attention-title">{{ selectedHead !== null ? `Layer ${selectedLayer ?? '—'} / head ${selectedHead}` : 'Attention pattern' }}</h2>
              </div>
              <span
                v-if="headDetail"
                class="provenance-badge provenance-badge--compact"
                :class="`provenance-badge--${headProvenance}`"
                :title="provenanceTitle(headProvenance)"
              >
                <span class="provenance-badge__dot" aria-hidden="true" />
                {{ headProvenance }}
              </span>
            </div>

            <div v-if="headLoading" class="loading-state loading-state--result" role="status">
              <span class="spinner" aria-hidden="true" />
              <span>Reading the selected attention head…</span>
            </div>
            <div v-else-if="headError" class="inline-error" role="alert">{{ headError }}</div>
            <div v-else-if="attentionMatrix.length" class="attention-content">
              <div class="attention-meta">
                <span>{{ attentionTokens.length ? `${attentionTokens.length} token positions` : 'Token labels unavailable' }}</span>
                <span v-if="headDetail?.id">{{ headDetail.id }}</span>
              </div>
              <div
                class="heatmap-wrap"
                role="img"
                :aria-label="`Attention values for layer ${selectedLayer ?? 'unknown'}, head ${selectedHead ?? 'unknown'}`"
              >
                <div
                  class="heatmap-grid"
                  :style="{ gridTemplateColumns: `repeat(${attentionMatrix[0]?.length ?? 1}, minmax(18px, 1fr))` }"
                >
                  <span
                    v-for="(cell, cellIndex) in flattenedAttentionCells"
                    :key="`cell-${cellIndex}`"
                    class="heatmap-cell"
                    :style="{ backgroundColor: attentionColor(cell) }"
                    :title="attentionCellTitle(cell, cellIndex)"
                    aria-hidden="true"
                  />
                </div>
              </div>
              <div v-if="attentionTokens.length" class="heatmap-labels" aria-hidden="true">
                <span v-for="(token, index) in attentionTokens" :key="`label-${index}-${token}`">
                  {{ formatToken(token) }}
                </span>
              </div>
              <dl v-if="headMetaRows.length" class="metadata-list metadata-list--compact">
                <div v-for="row in headMetaRows" :key="row.label" class="metadata-row">
                  <dt>{{ row.label }}</dt>
                  <dd>{{ row.value }}</dd>
                </div>
              </dl>
            </div>
            <div v-else class="empty-state empty-state--result">
              <strong>No attention pattern selected.</strong>
              <span>Run a prompt, then choose a layer and head. Head 0 loads by itself after each run.</span>
            </div>
          </section>

          <section class="panel analysis-panel" aria-labelledby="analysis-title">
            <div class="panel-heading panel-heading--compact">
              <div>
                <span class="panel-kicker">Activation surfaces</span>
                <h2 id="analysis-title">Layer evidence</h2>
              </div>
              <div
                v-if="activationDetail || layerActivationDetail"
                class="provenance-cluster"
                role="group"
                aria-label="Activation provenance"
              >
                <span
                  v-if="activationDetail"
                  class="provenance-badge provenance-badge--compact"
                  :class="`provenance-badge--${activationProvenance}`"
                  :title="provenanceTitle(activationProvenance)"
                >
                  <span class="provenance-badge__dot" aria-hidden="true" />
                  {{ activationProvenance }} cache
                </span>
                <span
                  v-if="layerActivationDetail"
                  class="provenance-badge provenance-badge--compact"
                  :class="`provenance-badge--${layerActivationProvenance}`"
                  :title="provenanceTitle(layerActivationProvenance)"
                >
                  <span class="provenance-badge__dot" aria-hidden="true" />
                  {{ layerActivationProvenance }} tensors
                </span>
              </div>
            </div>
            <div class="analysis-actions" role="group" aria-label="Activation and logit analysis actions">
              <button class="button button--secondary" type="button" :disabled="!isGpt2Model || activationsLoading" @click="loadActivations()">
                {{ activationsLoading ? 'Reading…' : 'Refresh cache shapes' }}
              </button>
              <button class="button button--secondary" type="button" :disabled="!isGpt2Model || layerActivationsLoading || !selectedLayerIsValid" @click="loadLayerActivations">
                {{ layerActivationsLoading ? 'Loading…' : 'Load layer tensors' }}
              </button>
              <button class="button button--secondary" type="button" :disabled="!isGpt2Model || lensLoading" @click="loadLogitLens">
                {{ lensLoading ? 'Computing…' : 'Run logit lens' }}
              </button>
            </div>
            <p v-if="activationDetail || layerActivationDetail" class="provenance-note">{{ activationProvenanceNote }}</p>

            <div v-if="activationsLoading || layerActivationsLoading" class="loading-state loading-state--result" role="status">
              <span class="spinner" aria-hidden="true" />
              <span>Requesting live activation evidence…</span>
            </div>
            <div v-else-if="activationError || layerActivationError" class="inline-error" role="alert">
              {{ activationError || layerActivationError }}
            </div>
            <div v-else-if="activationSummaryRows.length || layerActivationRows.length" class="activation-content">
              <dl v-if="activationSummaryRows.length" class="shape-list">
                <div v-for="row in activationSummaryRows" :key="row.label" class="shape-row">
                  <dt>{{ row.label }}</dt>
                  <dd>{{ row.value }}</dd>
                </div>
              </dl>
              <div v-if="layerActivationRows.length" class="activation-table-wrap">
                <div class="subheading">Layer {{ selectedLayer }} · first returned positions</div>
                <table class="data-table">
                  <caption class="sr-only">Returned residual and MLP activation values</caption>
                  <thead>
                    <tr>
                      <th scope="col">Token</th>
                      <th scope="col">Residual[0]</th>
                      <th scope="col">MLP[0]</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="row in layerActivationRows" :key="`activation-${row.index}-${row.token}`">
                      <th scope="row">{{ formatToken(row.token) }}</th>
                      <td>{{ formatNumber(row.resid) }}</td>
                      <td>{{ formatNumber(row.mlp) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div v-else class="empty-state empty-state--result">
              <strong>No activation evidence loaded.</strong>
              <span>Run a prompt to fill this panel. Shapes, tensors, and the lens load by themselves; the buttons refresh them.</span>
            </div>

            <div class="lens-section" :aria-busy="lensLoading">
              <div class="subheading-row">
                <div class="subheading">Logit lens</div>
                <span v-if="lensDetail" class="provenance-note provenance-note--inline">{{ provenanceTitle(lensProvenance) }}</span>
                <span v-if="lensDetail?.method" class="field-help">{{ lensDetail.method }}</span>
              </div>
              <div v-if="lensLoading" class="loading-state" role="status">
                <span class="spinner" aria-hidden="true" />
                <span>Projecting the returned hidden states…</span>
              </div>
              <div v-else-if="lensError" class="inline-error" role="alert">{{ lensError }}</div>
              <ol v-else-if="lensRows.length" class="lens-list" aria-label="Logit lens layers">
                <li v-for="row in lensRows" :key="`lens-${row.layer}`" class="lens-row">
                  <button type="button" class="lens-row__layer" @click="selectLayer(row.layer)">
                    L{{ row.layer }}
                  </button>
                  <span class="lens-row__token">{{ row.topToken ? formatToken(row.topToken) : 'No top token returned' }}</span>
                  <span v-if="row.tokens.length" class="lens-row__tokens">{{ row.tokens.join(' · ') }}</span>
                </li>
              </ol>
              <p v-else class="empty-state empty-state--inline">No logit-lens result yet. It runs by itself after each prompt.</p>
            </div>
          </section>
        </div>

        <aside class="panel component-panel" aria-labelledby="component-title">
          <div class="panel-heading panel-heading--compact">
            <div>
              <span class="panel-kicker">Component inspector</span>
              <h2 id="component-title">Selection detail</h2>
            </div>
            <span
              v-if="componentProvenance !== 'unavailable'"
              class="provenance-badge provenance-badge--compact"
              :class="`provenance-badge--${componentProvenance}`"
              :title="provenanceTitle(componentProvenance)"
            >
              <span class="provenance-badge__dot" aria-hidden="true" />
              {{ componentProvenance }}
            </span>
          </div>

          <div class="inspector-tabs" role="tablist" aria-label="Component detail type" @keydown="onInspectorTabKeydown">
            <button
              v-for="tab in inspectorTabs"
              :key="tab.id"
              class="inspector-tab"
              type="button"
              role="tab"
              :aria-selected="inspectorTab === tab.id"
              :aria-controls="inspectorTab === tab.id ? `component-panel-${tab.id}` : undefined"
              :tabindex="inspectorTab === tab.id ? 0 : -1"
              @click="inspectorTab = tab.id"
            >
              {{ tab.label }}
            </button>
          </div>

          <div
            v-if="inspectorTab === 'layer'"
            id="component-panel-layer"
            class="component-detail"
            role="tabpanel"
            aria-labelledby="component-title"
          >
            <div v-if="layerLoading" class="loading-state loading-state--side" role="status">
              <span class="spinner" aria-hidden="true" />
              <span>Loading layer detail…</span>
            </div>
            <div v-else-if="layerError" class="inline-error" role="alert">{{ layerError }}</div>
            <div v-else-if="layerDetail" class="detail-content">
              <div class="detail-title">Layer {{ layerDetail.layer ?? selectedLayer }}</div>
              <p class="detail-description">{{ layerDetail.path || 'Path metadata was not returned.' }}</p>
              <dl v-if="layerMetaRows.length" class="metadata-list">
                <div v-for="row in layerMetaRows" :key="row.label" class="metadata-row">
                  <dt>{{ row.label }}</dt>
                  <dd>{{ row.value }}</dd>
                </div>
              </dl>
              <div v-if="layerNeuronPreview.length" class="detail-list">
                <div class="subheading">Returned active neurons</div>
                <button
                  v-for="neuron in layerNeuronPreview"
                  :key="`layer-neuron-${neuron.index}`"
                  class="detail-list__row"
                  type="button"
                  @click="selectNeuron(neuron.index)"
                >
                  <span>N{{ neuron.index }}</span>
                  <span v-if="neuron.token">{{ formatToken(neuron.token) }}</span>
                  <span v-if="neuron.activation !== undefined">{{ formatNumber(neuron.activation) }}</span>
                </button>
              </div>
            </div>
            <div v-else class="empty-state empty-state--side">
              <strong>No layer detail.</strong>
              <span>Load GPT-2 to read layer detail. Layer 0 loads by itself.</span>
            </div>
          </div>

          <div
            v-else-if="inspectorTab === 'head'"
            id="component-panel-head"
            class="component-detail"
            role="tabpanel"
            aria-labelledby="component-title"
          >
            <div v-if="headLoading" class="loading-state loading-state--side" role="status">
              <span class="spinner" aria-hidden="true" />
              <span>Loading head detail…</span>
            </div>
            <div v-else-if="headError" class="inline-error" role="alert">{{ headError }}</div>
            <div v-else-if="headDetail" class="detail-content">
              <div class="detail-title">{{ headDetail.id || `L${selectedLayer ?? '—'}H${selectedHead ?? '—'}` }}</div>
              <p class="detail-description">{{ headDetail.description || headDetail.path || 'Head metadata was not returned.' }}</p>
              <dl v-if="headMetaRows.length" class="metadata-list">
                <div v-for="row in headMetaRows" :key="`head-detail-${row.label}`" class="metadata-row">
                  <dt>{{ row.label }}</dt>
                  <dd>{{ row.value }}</dd>
                </div>
              </dl>
            </div>
            <div v-else class="empty-state empty-state--side">
              <strong>No head detail.</strong>
              <span>Choose a head to read its pattern. Head 0 loads by itself after a prompt run.</span>
            </div>
          </div>

          <div
            v-else
            id="component-panel-neuron"
            class="component-detail"
            role="tabpanel"
            aria-labelledby="component-title"
          >
            <div v-if="neuronLoading" class="loading-state loading-state--side" role="status">
              <span class="spinner" aria-hidden="true" />
              <span>Loading neuron detail…</span>
            </div>
            <div v-else-if="neuronError" class="inline-error" role="alert">{{ neuronError }}</div>
            <div v-else-if="neuronDetail" class="detail-content">
              <div class="detail-title">{{ neuronDetail.id || `L${selectedLayer ?? '—'}.N${selectedNeuron ?? '—'}` }}</div>
              <p class="detail-description">{{ neuronDetail.description || neuronDetail.path || 'Neuron metadata was not returned.' }}</p>
              <dl v-if="neuronMetaRows.length" class="metadata-list">
                <div v-for="row in neuronMetaRows" :key="`neuron-meta-${row.label}`" class="metadata-row">
                  <dt>{{ row.label }}</dt>
                  <dd>{{ row.value }}</dd>
                </div>
              </dl>

              <div v-if="neuronTokenRows.length" class="detail-list detail-list--table">
                <div class="subheading">Returned token activations</div>
                <div v-for="row in neuronTokenRows" :key="`neuron-token-${row.index}`" class="neuron-token-row">
                  <code>{{ formatToken(row.token) }}</code>
                  <span>{{ formatNumber(row.activation) }}</span>
                  <span v-if="row.preActivation !== undefined" class="field-help">pre {{ formatNumber(row.preActivation) }}</span>
                </div>
              </div>

              <div v-if="neuronWeightRows.length" class="weight-list">
                <div class="subheading">Returned top input weights</div>
                <div v-for="row in neuronWeightRows" :key="`weight-${row.dim}`" class="weight-row">
                  <span>D{{ row.dim }}</span>
                  <span class="weight-track" aria-hidden="true"><span :style="{ width: `${row.width}%` }" /></span>
                  <code>{{ formatNumber(row.weight) }}</code>
                </div>
              </div>
            </div>
            <div v-else class="empty-state empty-state--side">
              <strong>No neuron detail.</strong>
              <span>Type a neuron index, or run a prompt and pick from the most active neurons.</span>
            </div>
          </div>
        </aside>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { api } from '../services/api';

const props = defineProps<{
  modelName?: string;
}>();

type JsonRecord = Record<string, unknown>;
type Provenance = 'live' | 'seeded' | 'reference' | 'unavailable';
type RuntimeState = 'checking' | 'online' | 'offline';
type PromptMode = 'gpt2' | 'infer';
type InspectorTab = 'layer' | 'head' | 'neuron';

interface MetadataRow {
  label: string;
  value: string;
}

interface PredictionRow {
  token: string;
  logit?: number;
  probability?: number;
}

interface NeuronOption {
  index: number;
  token?: string;
  activation?: number;
}

interface AttentionRow {
  values: Array<number | null>;
}

const models = ref<string[]>([]);
const selectedModel = ref('');
const loadedModel = ref('');
const modelProvenance = ref<Provenance>('unavailable');
const modelFieldProvenance = ref<Record<string, string>>({});
const modelLoadNote = ref('');
const modelLoadError = ref('');
const modelLoadState = ref<'idle' | 'loading' | 'loaded' | 'error'>('idle');
const catalogProvenance = ref<Provenance>('unavailable');
const catalogFieldProvenance = ref<Record<string, string>>({});
const catalogError = ref('');
const catalogLoading = ref(false);
const runtimeState = ref<RuntimeState>('checking');
const browserOnline = ref(typeof navigator === 'undefined' ? true : navigator.onLine);

const architecture = ref<JsonRecord | null>(null);
const architectureProvenance = ref<Provenance>('unavailable');
const architectureLoading = ref(false);
const architectureError = ref('');

const prompt = ref('The capital of France is');

const autoPrompt = ref(true);
const freshPromptNote = ref('');
const freshPromptState = ref<'idle' | 'loading'>('idle');
let freshPromptTimer: ReturnType<typeof setInterval> | null = null;
const runMode = ref<PromptMode>('gpt2');
const promptState = ref<'idle' | 'loading' | 'ready' | 'error'>('idle');
const promptError = ref('');
const promptResult = ref<JsonRecord | null>(null);
const promptSource = ref<PromptMode | null>(null);
const promptProvenance = ref<Provenance>('unavailable');

const selectedLayer = ref<number | null>(null);
const selectedHead = ref<number | null>(null);
const selectedNeuron = ref<number | null>(null);
const neuronIndexInput = ref<number | ''>('');
const layerDetail = ref<JsonRecord | null>(null);
const layerProvenance = ref<Provenance>('unavailable');
const layerLoading = ref(false);
const layerError = ref('');

const headDetail = ref<JsonRecord | null>(null);
const headProvenance = ref<Provenance>('unavailable');
const headLoading = ref(false);
const headError = ref('');

const neuronDetail = ref<JsonRecord | null>(null);
const neuronProvenance = ref<Provenance>('unavailable');
const neuronLoading = ref(false);
const neuronError = ref('');

const activationDetail = ref<JsonRecord | null>(null);
const layerActivationDetail = ref<JsonRecord | null>(null);
const activationsLoading = ref(false);
const layerActivationsLoading = ref(false);
const activationError = ref('');
const layerActivationError = ref('');
const activationProvenance = ref<Provenance>('unavailable');
const layerActivationProvenance = ref<Provenance>('unavailable');

const lensDetail = ref<JsonRecord | null>(null);
const lensProvenance = ref<Provenance>('unavailable');
const lensLoading = ref(false);
const lensError = ref('');

const inspectorTab = ref<InspectorTab>('layer');

const inspectorTabs: Array<{ id: InspectorTab; label: string }> = [
  { id: 'layer', label: 'Layer' },
  { id: 'head', label: 'Head' },
  { id: 'neuron', label: 'Neuron' },
];

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asText(value: unknown): string | undefined {
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return undefined;
}

function asNumber(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function asIndex(value: unknown): number | undefined {
  const parsed = asNumber(value);
  return parsed !== undefined && Number.isInteger(parsed) && parsed >= 0 ? parsed : undefined;
}

function firstText(source: JsonRecord | null | undefined, keys: string[]): string | undefined {
  if (!source) return undefined;
  for (const key of keys) {
    const value = asText(source[key]);
    if (value !== undefined && value.trim()) return value;
  }
  return undefined;
}

function firstNumber(source: JsonRecord | null | undefined, keys: string[]): number | undefined {
  if (!source) return undefined;
  for (const key of keys) {
    const value = asNumber(source[key]);
    if (value !== undefined) return value;
  }
  return undefined;
}

function firstRecord(source: JsonRecord | null | undefined, keys: string[]): JsonRecord | null {
  if (!source) return null;
  for (const key of keys) {
    if (isRecord(source[key])) return source[key];
  }
  return null;
}

function firstArray(source: JsonRecord | null | undefined, keys: string[]): unknown[] {
  if (!source) return [];
  for (const key of keys) {
    if (Array.isArray(source[key])) return source[key];
  }
  return [];
}

function responseError(value: unknown, label: string): string | null {
  if (!isRecord(value)) return `${label} returned an unreadable response.`;
  const explicit = firstText(value, ['error', 'message', 'detail']);
  if (explicit) return explicit;
  const status = firstText(value, ['status'])?.toLowerCase();
  if (status && ['error', 'failed', 'unavailable'].includes(status)) {
    return `${label} returned status “${status}” without a readable result.`;
  }
  return null;
}

function requireResponse(value: unknown, label: string): JsonRecord {
  const error = responseError(value, label);
  if (error) throw new Error(error);
  if (!isRecord(value)) throw new Error(`${label} returned an unreadable response.`);
  return value;
}

function errorText(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  return asText(error) ?? 'Unknown runtime error.';
}

function provenanceFrom(value: unknown, fallback: Provenance = 'unavailable'): Provenance {
  const raw = isRecord(value) ? asText(value.provenance)?.toLowerCase() : undefined;
  if (raw === 'live' || raw === 'seeded' || raw === 'reference' || raw === 'unavailable') return raw;
  return fallback;
}

function fieldProvenanceOf(value: unknown): Record<string, string> {
  if (!isRecord(value)) return {};
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, asText(item) ?? 'unavailable']));
}

function formatFieldProvenance(value: Record<string, string> | undefined): string {
  const entries = Object.entries(value ?? {});
  return entries.length ? entries.map(([key, item]) => `${key}: ${item}`).join(' · ') : 'Unavailable';
}

function provenanceTitle(provenance: Provenance): string {
  switch (provenance) {
    case 'live': return 'Backend reported live model data.';
    case 'seeded': return 'Backend reported deterministic seeded data, not live weights.';
    case 'reference': return 'Backend returned descriptive reference metadata; no live measurement is claimed.';
    case 'unavailable': return 'No verified model data is available.';
  }
}

function formatToken(value: string): string {
  return value.replaceAll('Ġ', '␣').replaceAll('Ċ', '⏎').replaceAll('\n', '⏎');
}

function formatNumber(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return value.toFixed(digits);
}

function formatProbability(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(2)}%`;
}

function formatShape(value: unknown): string | undefined {
  if (!Array.isArray(value)) return undefined;
  const dimensions = value.map((item) => asNumber(item)).filter((item): item is number => item !== undefined);
  return dimensions.length ? `[${dimensions.join(' × ')}]` : undefined;
}

function extractModels(value: unknown): string[] {
  const source = isRecord(value) ? value.models : value;
  if (!Array.isArray(source)) return [];
  const names = source
    .map((item) => {
      if (typeof item === 'string') return item;
      if (isRecord(item)) return firstText(item, ['name', 'model_name', 'id']);
      return undefined;
    })
    .filter((name): name is string => Boolean(name?.trim()));
  return [...new Set(names)];
}

function extractTokens(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .map((item) => {
        if (typeof item === 'string') return item;
        if (isRecord(item)) return firstText(item, ['text', 'token', 'token_str', 'value']);
        return undefined;
      })
      .filter((token): token is string => token !== undefined);
  }
  if (!isRecord(value)) return [];
  for (const key of ['str_tokens', 'tokens', 'token_strings']) {
    const tokens = value[key];
    if (Array.isArray(tokens)) return extractTokens(tokens);
  }
  return [];
}

function extractPredictions(value: JsonRecord | null): PredictionRow[] {
  const candidates = firstArray(value, ['top5', 'top_predictions', 'predictions', 'top16']);
  return candidates
    .map((item): PredictionRow | null => {
      if (typeof item === 'string') return { token: item };
      if (!isRecord(item)) return null;
      const token = firstText(item, ['token', 'token_str', 'text', 'value']);
      if (!token) return null;
      return {
        token,
        logit: firstNumber(item, ['logit', 'logit_value']),
        probability: firstNumber(item, ['probability', 'prob', 'score']),
      };
    })
    .filter((item): item is PredictionRow => item !== null)
    .slice(0, 8);
}

function extractNextToken(value: JsonRecord | null, source: PromptMode | null, tokens: string[]): { token: string; source: string } | null {
  const direct = firstText(value, ['next_token', 'nextToken', 'next_token_text']);
  if (direct) return { token: direct, source: 'explicit backend field' };
  const top = firstRecord(value, ['top1', 'top_prediction']);
  const topToken = firstText(top, ['token', 'token_str', 'text', 'value']);
  if (topToken) return { token: topToken, source: 'top-1 field' };
  if (source === 'infer' && tokens.length) {
    return { token: tokens[tokens.length - 1], source: 'last token returned by infer' };
  }
  return null;
}

function extractMatrix(value: unknown, maxRows = 12, maxColumns = 12): AttentionRow[] {
  if (!Array.isArray(value)) return [];
  return value.slice(0, maxRows).map((row) => {
    if (!Array.isArray(row)) return { values: [] };
    return {
      values: row.slice(0, maxColumns).map((cell) => asNumber(cell) ?? null),
    };
  });
}

function numberAt(matrix: unknown, row: number, column: number): number | undefined {
  if (!Array.isArray(matrix)) return undefined;
  const selectedRow = matrix[row];
  if (!Array.isArray(selectedRow)) return undefined;
  return asNumber(selectedRow[column]);
}

function isGpt2Name(value: string | null | undefined): boolean {
  return /gpt-?2|distilgpt/i.test(value ?? '');
}

function sameModel(left: string | null | undefined, right: string | null | undefined): boolean {
  if (!left || !right) return false;
  return left === right || (isGpt2Name(left) && isGpt2Name(right));
}

const isOffline = computed(() => !browserOnline.value || runtimeState.value === 'offline');
const isBusy = computed(() => catalogLoading.value || modelLoadState.value === 'loading' || architectureLoading.value || promptState.value === 'loading' || headLoading.value || layerLoading.value || neuronLoading.value || activationsLoading.value || layerActivationsLoading.value || lensLoading.value);
const runtimeLabel = computed(() => {
  if (!browserOnline.value) return 'Browser offline';
  if (runtimeState.value === 'offline') return 'Backend unavailable';
  if (runtimeState.value === 'checking') return 'Checking runtime';
  return 'Runtime reachable';
});
const offlineNotice = computed(() => {
  if (!browserOnline.value) return 'Reconnect this browser to read model data and run inference.';
  if (runtimeState.value !== 'offline') return '';
  return 'The MECH runtime at localhost:8000 is not responding. Existing returned data stays labeled; no replacement data is generated.';
});
const isGpt2Model = computed(() => isGpt2Name(selectedModel.value) || isGpt2Name(loadedModel.value));
const canLoadModel = computed(() => Boolean(selectedModel.value) && !isOffline.value && modelLoadState.value !== 'loading');
const canRunPrompt = computed(() => Boolean(prompt.value.trim()) && Boolean(selectedModel.value) && !isOffline.value && promptState.value !== 'loading');
const selectedLayerIsValid = computed(() => selectedLayer.value !== null && selectedLayer.value >= 0);
const resultModelName = computed(() => firstText(promptResult.value, ['model_name', 'model', 'modelName']));
const resultTokens = computed(() => extractTokens(promptResult.value));
const resultTokenCount = computed(() => resultTokens.value.length);
const predictions = computed(() => extractPredictions(promptResult.value));
const nextTokenResult = computed(() => extractNextToken(promptResult.value, promptSource.value, resultTokens.value));
const nextToken = computed(() => nextTokenResult.value?.token ?? '');
const nextTokenSource = computed(() => nextTokenResult.value?.source ?? '');
const promptProvenanceNote = computed(() => provenanceTitle(promptProvenance.value));
const promptFieldProvenance = computed(() => fieldProvenanceOf(promptResult.value?.field_provenance));
const displayActivationProvenance = computed<Provenance>(() => (
  activationDetail.value ? activationProvenance.value : layerActivationProvenance.value
));
const activationProvenanceNote = computed(() => provenanceTitle(displayActivationProvenance.value));

const architectureLayers = computed<JsonRecord[]>(() => {
  const layers = architecture.value?.layers;
  return Array.isArray(layers) ? layers.filter(isRecord) : [];
});
const layerCount = computed(() => {
  const explicit = firstNumber(architecture.value, ['n_layers', 'num_layers', 'layer_count']);
  if (explicit !== undefined && Number.isInteger(explicit) && explicit >= 0) return explicit;
  return architectureLayers.value.length;
});
const layerOptions = computed(() => Array.from({ length: Math.max(0, layerCount.value) }, (_, index) => index));
const selectedLayerRecord = computed(() => architectureLayers.value.find((layer) => asIndex(layer.layer_index) === selectedLayer.value) ?? null);
const headCount = computed(() => {
  const explicit = firstNumber(selectedLayerRecord.value, ['num_attention_heads', 'n_heads', 'num_heads']);
  if (explicit !== undefined && Number.isInteger(explicit) && explicit >= 0) return explicit;
  const architectureExplicit = firstNumber(architecture.value, ['n_heads', 'num_attention_heads', 'head_count']);
  if (architectureExplicit !== undefined && Number.isInteger(architectureExplicit) && architectureExplicit >= 0) return architectureExplicit;
  const heads = firstArray(selectedLayerRecord.value, ['attention_heads', 'attention_heads_preview']);
  return heads.length;
});
const headOptions = computed(() => {
  const layerHeads = firstArray(layerDetail.value, ['attention_heads', 'attention_heads_preview']);
  const architectureHeads = firstArray(selectedLayerRecord.value, ['attention_heads', 'attention_heads_preview']);
  const source = layerHeads.length ? layerHeads : architectureHeads;
  const indices = source
    .map((head) => isRecord(head) ? asIndex(head.head_index ?? head.head) : asIndex(head))
    .filter((index): index is number => index !== undefined);
  if (indices.length) return [...new Set(indices)];
  return Array.from({ length: Math.max(0, headCount.value) }, (_, index) => index);
});
const maxNeuronIndex = computed(() => {
  const explicit = firstNumber(selectedLayerRecord.value, ['num_mlp_neurons', 'd_mlp', 'mlp_width', 'neuron_count']);
  if (explicit !== undefined && Number.isInteger(explicit) && explicit > 0) return explicit;
  return firstNumber(layerDetail.value, ['num_mlp_neurons', 'd_mlp', 'mlp_width', 'neuron_count']);
});
const maxNeuronInput = computed(() => {
  const count = maxNeuronIndex.value;
  return count !== undefined && count > 0 ? count - 1 : undefined;
});
const canInspectNeuron = computed(() => isGpt2Model.value && neuronIndexInput.value !== '' && !neuronLoading.value);
const neuronOptions = computed<NeuronOption[]>(() => {
  const source = firstArray(layerDetail.value, ['top_active_neurons', 'mlp_neurons_preview']);
  return source
    .map((item) => {
      if (!isRecord(item)) return null;
      const index = asIndex(item.neuron_index ?? item.index);
      if (index === undefined) return null;
      return {
        index,
        token: firstText(item, ['top_token', 'token', 'strongest_token']),
        activation: firstNumber(item, ['activation', 'max_activation', 'mean_activation']),
      };
    })
    .filter((item): item is NeuronOption => item !== null)
    .slice(0, 32);
});
const layerNeuronPreview = computed<NeuronOption[]>(() => neuronOptions.value.slice(0, 16));

const architectureRows = computed<MetadataRow[]>(() => {
  if (!architecture.value) return [];
  const rows: Array<[string, string | undefined]> = [
    ['Model', firstText(architecture.value, ['model_name', 'model', 'name'])],
    ['Family', firstText(architecture.value, ['model_type', 'family', 'architecture'])],
    ['Layers', formatNumber(firstNumber(architecture.value, ['n_layers', 'num_layers']), 0)],
    ['Heads', formatNumber(firstNumber(architecture.value, ['n_heads', 'num_attention_heads']), 0)],
    ['d_model', formatNumber(firstNumber(architecture.value, ['d_model', 'hidden_size']), 0)],
    ['d_mlp', formatNumber(firstNumber(architecture.value, ['d_mlp', 'intermediate_size']), 0)],
    ['d_head', formatNumber(firstNumber(architecture.value, ['d_head', 'head_dim']), 0)],
    ['Vocabulary', formatNumber(firstNumber(architecture.value, ['vocab_size', 'vocabulary_size']), 0)],
    ['Parameters', firstText(architecture.value, ['n_params_human']) ?? formatNumber(firstNumber(architecture.value, ['n_params', 'num_parameters']), 0)],
    ['Device', firstText(architecture.value, ['device'])],
    ['Cache', firstText(architecture.value, ['cached_prompt']) ? 'prompt cached' : undefined],
  ];
  return rows.filter((row): row is [string, string] => Boolean(row[1] && row[1] !== '—'));
});

const attentionTokens = computed(() => {
  const headTokens = extractTokens(headDetail.value);
  return headTokens.length ? headTokens : resultTokens.value;
});
const attentionMatrix = computed(() => {
  const matrix = headDetail.value?.matrix ?? headDetail.value?.attention_matrix;
  return extractMatrix(matrix);
});
const flattenedAttentionCells = computed<Array<number | null>>(() => attentionMatrix.value.flatMap((row) => row.values));
const attentionMax = computed(() => {
  const values = flattenedAttentionCells.value.filter((value): value is number => value !== null);
  return values.length ? Math.max(...values) : 0;
});
const attentionColor = (cell: number | null): string => {
  if (cell === null || !Number.isFinite(cell)) return 'var(--surface-2)';
  const ratio = attentionMax.value ? Math.max(0, Math.min(1, cell / attentionMax.value)) : 0;
  return `rgba(37, 99, 235, ${0.08 + ratio * 0.78})`;
};
function attentionCellTitle(cell: number | null, cellIndex: number): string {
  if (cell === null) return 'No value';
  const columns = attentionMatrix.value[0]?.length ?? 1;
  return `row ${Math.floor(cellIndex / columns)}, column ${cellIndex % columns}: ${formatNumber(cell)}`;
}
const headMetaRows = computed<MetadataRow[]>(() => {
  if (!headDetail.value) return [];
  const rows: Array<[string, string | undefined]> = [
    ['ID', firstText(headDetail.value, ['id', 'label'])],
    ['Path', firstText(headDetail.value, ['path'])],
    ['d_head', formatNumber(firstNumber(headDetail.value, ['d_head', 'head_dim']), 0)],
    ['d_model', formatNumber(firstNumber(headDetail.value, ['d_model', 'hidden_size']), 0)],
    ['Pattern', headDetail.value.has_pattern === true ? 'returned' : undefined],
  ];
  return rows.filter((row): row is [string, string] => Boolean(row[1]));
});

const layerMetaRows = computed<MetadataRow[]>(() => {
  if (!layerDetail.value) return [];
  const rows: Array<[string, string | undefined]> = [
    ['Path', firstText(layerDetail.value, ['path', 'label'])],
    ['Heads', formatNumber(firstNumber(layerDetail.value, ['num_attention_heads', 'n_heads']), 0)],
    ['MLP neurons', formatNumber(firstNumber(layerDetail.value, ['num_mlp_neurons', 'd_mlp']), 0)],
    ['Residual dim', formatNumber(firstNumber(layerDetail.value, ['residual_stream_dim', 'd_model']), 0)],
    ['Parameters', formatNumber(firstNumber(layerDetail.value, ['n_params', 'num_parameters']), 0)],
    ['Activation cache', layerDetail.value.has_activations === true ? 'available' : layerDetail.value.has_activations === false ? 'not populated' : undefined],
  ];
  return rows.filter((row): row is [string, string] => Boolean(row[1] && row[1] !== '—'));
});

const neuronMetaRows = computed<MetadataRow[]>(() => {
  if (!neuronDetail.value) return [];
  const rows: Array<[string, string | undefined]> = [
    ['Component', firstText(neuronDetail.value, ['component'])],
    ['Path', firstText(neuronDetail.value, ['path'])],
    ['Bias', formatNumber(firstNumber(neuronDetail.value, ['bias']))],
    ['Input L2', formatNumber(firstNumber(neuronDetail.value, ['in_weight_l2']))],
    ['Output L2', formatNumber(firstNumber(neuronDetail.value, ['out_weight_l2']))],
    ['Activation cache', neuronDetail.value.has_activations === true ? 'available' : neuronDetail.value.has_activations === false ? 'not populated' : undefined],
  ];
  return rows.filter((row): row is [string, string] => Boolean(row[1] && row[1] !== '—'));
});

const neuronTokenRows = computed(() => {
  const rows = firstArray(neuronDetail.value, ['per_token_activations', 'token_activations']);
  return rows
    .map((item) => {
      if (!isRecord(item)) return null;
      const index = asIndex(item.token_index);
      if (index === undefined) return null;
      return {
        index,
        token: firstText(item, ['token', 'token_str', 'text']) ?? `token ${index}`,
        activation: firstNumber(item, ['activation', 'value']),
        preActivation: firstNumber(item, ['pre_activation', 'pre_activation_value']),
      };
    })
    .filter((item): item is { index: number; token: string; activation?: number; preActivation?: number } => item !== null);
});

const neuronWeightRows = computed(() => {
  const rows = firstArray(neuronDetail.value, ['top_input_weights_positive', 'top_input_weights_negative']);
  const maxWeight = Math.max(...rows.map((item) => Math.abs(asNumber(isRecord(item) ? item.weight : undefined) ?? 0)), 0.0001);
  return rows
    .map((item) => {
      if (!isRecord(item)) return null;
      const dim = asIndex(item.dim);
      const weight = firstNumber(item, ['weight', 'value']);
      if (dim === undefined || weight === undefined) return null;
      return { dim, weight, width: Math.min(100, Math.abs(weight) / maxWeight * 100) };
    })
    .filter((item): item is { dim: number; weight: number; width: number } => item !== null)
    .slice(0, 16);
});

const activationSummaryRows = computed<MetadataRow[]>(() => {
  const source = activationDetail.value ?? layerActivationDetail.value;
  if (!source) return [];
  const rows: Array<[string, string | undefined]> = [
    ['Prompt', firstText(source, ['prompt'])],
    ['Residual shape', formatShape(source.resid_shape)],
    ['Attention shape', formatShape(source.attn_shape)],
    ['MLP shape', formatShape(source.mlp_shape)],
    ['Residual L2', formatNumber(firstNumber(firstRecord(source, ['stats']), ['resid_l2']))],
    ['MLP mean', formatNumber(firstNumber(firstRecord(source, ['stats']), ['mlp_mean']))],
    ['MLP sparsity', formatNumber(firstNumber(firstRecord(source, ['stats']), ['mlp_sparsity']))],
  ];
  return rows.filter((row): row is [string, string] => Boolean(row[1] && row[1] !== '—'));
});

const layerActivationRows = computed(() => {
  const source = layerActivationDetail.value;
  if (!source) return [];
  const tokens = extractTokens(source.tokens);
  const resid = source.resid_post;
  const mlp = source.mlp_post;
  return tokens.slice(0, 12).map((token, index) => ({
    index,
    token,
    resid: numberAt(resid, index, 0),
    mlp: numberAt(mlp, index, 0),
  }));
});

const lensRows = computed(() => {
  const rows = firstArray(lensDetail.value, ['layers', 'logit_lens']);
  return rows
    .map((item) => {
      if (!isRecord(item)) return null;
      const layer = asIndex(item.layer ?? item.layer_index);
      if (layer === undefined) return null;
      const topK = firstArray(item, ['top_k_tokens', 'top_tokens', 'predictions']);
      const tokens = topK
        .map((token) => isRecord(token) ? firstText(token, ['token', 'token_str', 'text', 'value']) : asText(token))
        .filter((token): token is string => Boolean(token));
      return {
        layer,
        topToken: firstText(item, ['top_token', 'top_prediction']) ?? tokens[0],
        tokens,
      };
    })
    .filter((item): item is { layer: number; topToken?: string; tokens: string[] } => item !== null);
});

const componentProvenance = computed<Provenance>(() => {
  if (inspectorTab.value === 'head' && headDetail.value) return headProvenance.value;
  if (inspectorTab.value === 'neuron' && neuronDetail.value) return neuronProvenance.value;
  if (inspectorTab.value === 'layer' && layerDetail.value) return layerProvenance.value;
  return 'unavailable';
});
const architectureHint = computed(() => {
  if (!isGpt2Model.value) return 'Load a GPT-2-family model to expose layer, head, and neuron controls.';
  if (modelLoadState.value === 'idle') return 'Load the selected GPT-2 model to request architecture metadata.';
  return 'The runtime did not return usable architecture metadata.';
});

function resetPromptDependentState(): void {
  promptResult.value = null;
  promptSource.value = null;
  promptProvenance.value = 'unavailable';
  promptState.value = 'idle';
  promptError.value = '';
  activationDetail.value = null;
  layerActivationDetail.value = null;
  activationProvenance.value = 'unavailable';
  layerActivationProvenance.value = 'unavailable';
  activationError.value = '';
  layerActivationError.value = '';
  lensDetail.value = null;
  lensProvenance.value = 'unavailable';
  lensError.value = '';
  layerDetail.value = null;
  layerProvenance.value = 'unavailable';
  layerError.value = '';
  headDetail.value = null;
  headProvenance.value = 'unavailable';
  headError.value = '';
  neuronDetail.value = null;
  neuronProvenance.value = 'unavailable';
  neuronError.value = '';
}

function resetModelDependentState(): void {
  loadedModel.value = '';
  modelProvenance.value = 'unavailable';
  modelFieldProvenance.value = {};
  modelLoadState.value = 'idle';
  modelLoadNote.value = '';
  modelLoadError.value = '';
  architecture.value = null;
  architectureProvenance.value = 'unavailable';
  architectureError.value = '';
  selectedLayer.value = null;
  selectedHead.value = null;
  selectedNeuron.value = null;
  neuronIndexInput.value = '';
  resetPromptDependentState();
}

function selectModel(model: string): void {
  if (selectedModel.value === model && sameModel(loadedModel.value, model)) return;
  selectedModel.value = model;
  if (loadedModel.value && !sameModel(loadedModel.value, model)) resetModelDependentState();
  modelLoadState.value = sameModel(loadedModel.value, model) ? 'loaded' : 'idle';
  modelLoadError.value = '';
  modelLoadNote.value = '';
  resetPromptDependentState();
}

function onModelSelectionChange(event: Event): void {
  const value = (event.target as HTMLSelectElement).value;
  if (value) selectModel(value);
}

async function loadCatalog(): Promise<void> {
  if (!browserOnline.value) {
    runtimeState.value = 'offline';
    catalogError.value = 'The browser is offline.';
    return;
  }
  catalogLoading.value = true;
  catalogError.value = '';
  catalogFieldProvenance.value = {};
  try {
    const response = requireResponse(await api.listModels(), 'Model registry');
    models.value = extractModels(response);
    catalogProvenance.value = provenanceFrom(response, 'reference');
    catalogFieldProvenance.value = fieldProvenanceOf(response.field_provenance);
    runtimeState.value = 'online';
    const requestedModel = props.modelName?.trim();
    if (!selectedModel.value && requestedModel && models.value.includes(requestedModel)) {
      selectedModel.value = requestedModel;
    }
    if (!selectedModel.value && models.value.length) selectedModel.value = models.value[0];
    if (!models.value.includes(selectedModel.value)) {
      selectedModel.value = models.value[0] ?? '';
      if (loadedModel.value) resetModelDependentState();
    }
    if (!models.value.length) catalogProvenance.value = 'unavailable';
  } catch (error) {
    runtimeState.value = 'offline';
    catalogProvenance.value = 'unavailable';
    catalogError.value = errorText(error);
  } finally {
    catalogLoading.value = false;
  }
}

async function loadSelectedModel(): Promise<void> {
  const model = selectedModel.value;
  if (!model) {
    modelLoadError.value = 'Choose a model from the registry first.';
    return;
  }
  if (isOffline.value) {
    modelLoadError.value = 'Reconnect before loading a model.';
    return;
  }

  modelLoadState.value = 'loading';
  modelLoadError.value = '';
  modelLoadNote.value = '';
  architectureError.value = '';
  try {
    const response = requireResponse(
      isGpt2Name(model) ? await api.gpt2Load(model) : await api.loadModel(model),
      `Model load (${model})`,
    );
    loadedModel.value = firstText(response, ['model_name', 'model', 'name']) ?? model;
    modelProvenance.value = provenanceFrom(response, 'reference');
    modelFieldProvenance.value = fieldProvenanceOf(response.field_provenance);
    modelLoadState.value = 'loaded';
    modelLoadNote.value = firstText(response, ['provenance_note', 'note']) ?? (modelProvenance.value === 'reference'
      ? 'The backend acknowledged the load without a live provenance marker.'
      : '');

    if (isGpt2Name(loadedModel.value) || isGpt2Name(model)) {
      await loadArchitecture();
    } else {
      architecture.value = null;
      architectureProvenance.value = 'unavailable';
      architectureError.value = 'The GPT-2 inspector is not available for this model.';
    }
  } catch (error) {
    modelLoadState.value = 'error';
    modelLoadError.value = errorText(error);
    modelProvenance.value = 'unavailable';
  }
}

async function loadArchitecture(): Promise<void> {
  if (!isGpt2Name(selectedModel.value) && !isGpt2Name(loadedModel.value)) {
    architecture.value = null;
    architectureProvenance.value = 'unavailable';
    architectureError.value = 'Load a GPT-2-family model to request architecture metadata.';
    return;
  }
  architectureLoading.value = true;
  architectureError.value = '';
  try {
    const response = requireResponse(await api.gpt2Architecture(), 'GPT-2 architecture');
    architecture.value = response;
    architectureProvenance.value = provenanceFrom(response, 'reference');
    const firstLayer = layerOptions.value[0];
    if (selectedLayer.value === null && firstLayer !== undefined) await selectLayer(firstLayer);
  } catch (error) {
    architecture.value = null;
    architectureProvenance.value = 'unavailable';
    architectureError.value = errorText(error);
  } finally {
    architectureLoading.value = false;
  }
}

function onPromptKeydown(event: KeyboardEvent): void {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    void runPrompt();
  }
}

function freshPromptTimestamp(): string {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

async function fetchFreshPrompt(manual: boolean): Promise<void> {
  if (freshPromptState.value === 'loading') return;
  if (promptState.value === 'loading') {
    if (manual) freshPromptNote.value = 'Wait for the running inference to finish, then try again.';
    return;
  }
  if (!browserOnline.value) {
    freshPromptNote.value = 'Reconnect this browser before requesting a model-created prompt.';
    return;
  }
  freshPromptState.value = 'loading';
  try {
    const response = requireResponse(await api.gpt2FreshPrompt(), 'Fresh prompt');
    const text = firstText(response, ['prompt', 'text']);
    const provenance = provenanceFrom(response, 'unavailable');
    if (!text || provenance !== 'live') {
      freshPromptNote.value = 'The model did not return a usable prompt; the previous text was kept.';
      return;
    }
    prompt.value = text;
    promptError.value = '';
    freshPromptNote.value = `New prompt created by GPT-2 (live) · ${freshPromptTimestamp()}`;
  } catch (error) {
    freshPromptNote.value = `Fresh prompt unavailable: ${errorText(error)}`;
  } finally {
    freshPromptState.value = 'idle';
  }
}

function startFreshPromptTimer(): void {
  stopFreshPromptTimer();
  freshPromptTimer = setInterval(() => {
    if (!autoPrompt.value || document.hidden) return;
    void fetchFreshPrompt(false);
  }, 60_000);
}

function stopFreshPromptTimer(): void {
  if (freshPromptTimer !== null) {
    clearInterval(freshPromptTimer);
    freshPromptTimer = null;
  }
}

async function runPrompt(mode?: PromptMode): Promise<void> {
  const selectedMode = mode ?? runMode.value;
  const text = prompt.value.trim();
  if (!text) {
    promptError.value = 'Enter a prompt before running inference.';
    promptState.value = 'error';
    return;
  }
  if (!selectedModel.value) {
    promptError.value = 'Select a model from the registry before running inference.';
    promptState.value = 'error';
    return;
  }
  if (isOffline.value) {
    promptError.value = 'Reconnect before running inference.';
    promptState.value = 'error';
    return;
  }
  if (promptState.value === 'loading') return;

  promptState.value = 'loading';
  promptError.value = '';
  promptResult.value = null;
  promptSource.value = null;
  promptProvenance.value = 'unavailable';
  resetInspectionResults();
  try {
    const response = requireResponse(
      selectedMode === 'infer'
        ? await api.infer(text, selectedModel.value)
        : await api.gpt2RunPrompt(text),
      selectedMode === 'infer' ? 'Generic inference' : 'GPT-2 prompt run',
    );
    promptResult.value = response;
    promptSource.value = selectedMode;
    promptProvenance.value = provenanceFrom(response, 'unavailable');
    promptState.value = 'ready';
    if (nextToken.value) prompt.value = `${text}${nextToken.value}`;
    if (isGpt2Model.value) await loadActivations(false);
    if (isGpt2Model.value) await runInspectionCascade();
  } catch (error) {
    promptState.value = 'error';
    promptError.value = errorText(error);
  } finally {
    if (promptState.value === 'loading') promptState.value = 'idle';
  }
}

let inspectionRunId = 0;

function resetInspectionResults(): void {
  inspectionRunId += 1;
  activationDetail.value = null;
  layerActivationDetail.value = null;
  activationProvenance.value = 'unavailable';
  layerActivationProvenance.value = 'unavailable';
  activationError.value = '';
  layerActivationError.value = '';
  lensDetail.value = null;
  lensProvenance.value = 'unavailable';
  lensError.value = '';
  layerDetail.value = null;
  layerProvenance.value = 'unavailable';
  layerError.value = '';
  headDetail.value = null;
  headProvenance.value = 'unavailable';
  headError.value = '';
  neuronDetail.value = null;
  neuronProvenance.value = 'unavailable';
  neuronError.value = '';
  selectedHead.value = null;
  selectedNeuron.value = null;
  neuronIndexInput.value = '';
}

function onInspectorTabKeydown(event: KeyboardEvent): void {
  if (!['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) return;
  const tabList = event.currentTarget as HTMLElement | null;
  if (!tabList) return;
  const tabs = Array.from(tabList.querySelectorAll<HTMLButtonElement>('[role="tab"]'));
  const activeIndex = tabs.indexOf(document.activeElement as HTMLButtonElement);
  if (activeIndex < 0 || !tabs.length) return;
  event.preventDefault();
  let nextIndex = activeIndex;
  if (event.key === 'ArrowRight') nextIndex = (activeIndex + 1) % tabs.length;
  if (event.key === 'ArrowLeft') nextIndex = (activeIndex - 1 + tabs.length) % tabs.length;
  if (event.key === 'Home') nextIndex = 0;
  if (event.key === 'End') nextIndex = tabs.length - 1;
  inspectorTab.value = inspectorTabs[nextIndex]?.id ?? inspectorTab.value;
  tabs[nextIndex]?.focus();
}

function onLayerChange(event: Event): void {
  const value = asIndex((event.target as HTMLSelectElement).value);
  if (value !== undefined) void selectLayer(value);
}

async function selectLayer(layer: number, opts?: { keepTab?: boolean }): Promise<void> {
  if (!Number.isInteger(layer) || layer < 0) return;
  selectedLayer.value = layer;
  selectedHead.value = null;
  selectedNeuron.value = null;
  neuronIndexInput.value = '';
  headDetail.value = null;
  neuronDetail.value = null;
  if (!isGpt2Model.value) return;

  layerLoading.value = true;
  layerError.value = '';
  try {
    const response = requireResponse(await api.gpt2Layer(layer), `Layer ${layer}`);
    layerDetail.value = response;
    layerProvenance.value = provenanceFrom(response, 'reference');
    const firstOption = neuronOptions.value[0];
    if (firstOption) {
      await selectNeuron(firstOption.index);
    } else {
      neuronIndexInput.value = 0;
      await selectNeuron(0);
    }
    if (!opts?.keepTab) inspectorTab.value = 'layer';
    else if (selectedNeuron.value === null && !neuronDetail.value) inspectorTab.value = 'layer';
  } catch (error) {
    layerDetail.value = null;
    layerProvenance.value = 'unavailable';
    layerError.value = errorText(error);
  } finally {
    layerLoading.value = false;
  }
}

function onHeadChange(event: Event): void {
  const value = asIndex((event.target as HTMLSelectElement).value);
  if (value !== undefined) void selectHead(value);
}

async function selectHead(head: number): Promise<void> {
  if (selectedLayer.value === null) {
    headError.value = 'Select a layer before selecting a head.';
    return;
  }
  selectedHead.value = head;
  neuronDetail.value = null;
  selectedNeuron.value = null;
  inspectorTab.value = 'head';
  headLoading.value = true;
  headError.value = '';
  try {
    const response = requireResponse(await api.gpt2AttentionHead(selectedLayer.value, head), `Attention head L${selectedLayer.value}H${head}`);
    headDetail.value = response;
    headProvenance.value = provenanceFrom(response, 'unavailable');
  } catch (error) {
    headDetail.value = null;
    headProvenance.value = 'unavailable';
    headError.value = errorText(error);
  } finally {
    headLoading.value = false;
  }
}

function onNeuronInputChange(): void {
  const value = asIndex(neuronIndexInput.value);
  if (value === undefined) {
    selectedNeuron.value = null;
    neuronIndexInput.value = '';
    return;
  }
  neuronIndexInput.value = value;
}

function inspectNeuronInput(): void {
  const value = asIndex(neuronIndexInput.value);
  if (value !== undefined) void selectNeuron(value);
}

async function selectNeuron(index: number): Promise<void> {
  if (!isGpt2Model.value) {
    neuronError.value = 'The neuron inspector is only available for GPT-2-family models.';
    return;
  }
  if (selectedLayer.value === null) {
    neuronError.value = 'Select a layer before selecting a neuron.';
    return;
  }
  if (maxNeuronInput.value !== undefined && index > maxNeuronInput.value) {
    neuronError.value = `Neuron index must be between 0 and ${maxNeuronInput.value}.`;
    return;
  }
  selectedNeuron.value = index;
  neuronIndexInput.value = index;
  inspectorTab.value = 'neuron';
  neuronLoading.value = true;
  neuronError.value = '';
  try {
    const response = requireResponse(
      await api.gpt2Neuron(selectedLayer.value, index, 'mlp', 16),
      `Neuron L${selectedLayer.value}N${index}`,
    );
    neuronDetail.value = response;
    neuronProvenance.value = provenanceFrom(response, 'reference');
  } catch (error) {
    neuronDetail.value = null;
    neuronProvenance.value = 'unavailable';
    neuronError.value = errorText(error);
  } finally {
    neuronLoading.value = false;
  }
}

async function loadActivations(silent = false): Promise<void> {
  if (!isGpt2Model.value) {
    if (!silent) activationError.value = 'The activation cache is only available for GPT-2-family models.';
    return;
  }
  activationsLoading.value = true;
  activationError.value = '';
  try {
    const response = requireResponse(await api.gpt2GetActivations(selectedLayer.value ?? undefined), 'GPT-2 activations');
    activationDetail.value = response;
    activationProvenance.value = provenanceFrom(response, 'unavailable');
    layerActivationError.value = '';
  } catch (error) {
    activationDetail.value = null;
    activationProvenance.value = 'unavailable';
    if (!silent) activationError.value = errorText(error);
  } finally {
    activationsLoading.value = false;
  }
}

async function loadLayerActivations(): Promise<void> {
  if (!isGpt2Model.value || selectedLayer.value === null) {
    layerActivationError.value = 'Load GPT-2 and select a layer before requesting layer tensors.';
    return;
  }
  layerActivationsLoading.value = true;
  layerActivationError.value = '';
  try {
    const response = requireResponse(
      await api.pythonCall('gpt2/layer_activations', {
        layer: selectedLayer.value,
        prompt: prompt.value.trim(),
      }),
      `Layer activations (layer ${selectedLayer.value})`,
    );
    layerActivationDetail.value = response;
    layerActivationProvenance.value = provenanceFrom(response, 'reference');
    activationError.value = '';
  } catch (error) {
    layerActivationDetail.value = null;
    layerActivationProvenance.value = 'unavailable';
    layerActivationError.value = errorText(error);
  } finally {
    layerActivationsLoading.value = false;
  }
}

async function loadLogitLens(): Promise<void> {
  if (!isGpt2Model.value) {
    lensError.value = 'The logit lens is only available for GPT-2-family models.';
    return;
  }
  lensLoading.value = true;
  lensError.value = '';
  try {
    const response = requireResponse(
      await api.pythonCall('gpt2/logit_lens_all', { prompt: prompt.value.trim() }),
      'Logit lens',
    );
    lensDetail.value = response;
    lensProvenance.value = provenanceFrom(response, 'reference');
  } catch (error) {
    lensDetail.value = null;
    lensProvenance.value = 'unavailable';
    lensError.value = errorText(error);
  } finally {
    lensLoading.value = false;
  }
}

async function runInspectionCascade(): Promise<void> {
  const runId = ++inspectionRunId;
  const targetLayer = selectedLayer.value ?? layerOptions.value[0];
  if (targetLayer === undefined) return;
  await selectLayer(targetLayer, { keepTab: true });
  if (runId !== inspectionRunId) return;
  if (layerDetail.value) {
    await selectHead(0);
    if (runId !== inspectionRunId) return;
    const topNeuron = neuronOptions.value[0];
    if (topNeuron) await selectNeuron(topNeuron.index);
    if (runId !== inspectionRunId) return;
  }
  if (runId !== inspectionRunId) return;
  await loadLayerActivations();
  if (runId !== inspectionRunId) return;
  await loadLogitLens();
}

function handleOnline(): void {
  browserOnline.value = true;
  if (runtimeState.value === 'offline') runtimeState.value = 'checking';
}

function handleOffline(): void {
  browserOnline.value = false;
  runtimeState.value = 'offline';
}

onMounted(() => {
  window.addEventListener('online', handleOnline);
  window.addEventListener('offline', handleOffline);
  void loadCatalog();
  void fetchFreshPrompt(false);
  startFreshPromptTimer();
});

onBeforeUnmount(() => {
  stopFreshPromptTimer();
  window.removeEventListener('online', handleOnline);
  window.removeEventListener('offline', handleOffline);
});
</script>

<style scoped>
.model-explorer {
  min-height: 100%;
  background: var(--surface, #ffffff);
  color: var(--text, #182230);
  color-scheme: light;
  font-family: var(--font, "IBM Plex Sans", "Segoe UI", sans-serif);
  font-size: 12px;
}

.model-explorer__frame {
  width: min(1480px, 100%);
  margin: 0 auto;
  padding: 18px;
}

.explorer-header,
.panel-heading,
.prompt-label-row,
.selector-label-row,
.subheading-row,
.result-meta,
.catalog-meta,
.model-load-controls,
.loaded-model,
.analysis-actions,
.next-token-card,
.attention-meta,
.metadata-row,
.shape-row,
.neuron-input-row {
  display: flex;
  align-items: center;
}

.explorer-header {
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 16px;
}

.explorer-header__identity {
  min-width: 0;
}

.explorer-kicker,
.panel-kicker,
.subheading,
.field-help,
.keyboard-hint,
.runtime-state,
.provenance-badge,
.model-option__state,
.loaded-model,
.catalog-meta,
.result-meta,
.shape-row dt,
.shape-row dd,
.field-group label,
.selector-block > label,
.selector-label-row label,
.empty-state span,
.inline-error,
.architecture-content .field-help,
.lens-row__tokens {
  font-family: var(--font-mono, "IBM Plex Mono", "Cascadia Code", monospace);
}

.explorer-kicker,
.panel-kicker {
  display: block;
  color: var(--text-muted, #647184);
  font-size: 10px;
  font-weight: 650;
  letter-spacing: 0.08em;
  line-height: 1.3;
  text-transform: uppercase;
}

.explorer-header h1 {
  margin: 3px 0 0;
  color: var(--text, #182230);
  font-size: clamp(22px, 2.4vw, 30px);
  font-weight: 720;
  letter-spacing: -0.035em;
  line-height: 1.1;
}

.explorer-header p {
  max-width: 720px;
  margin: 7px 0 0;
  color: var(--text-muted, #647184);
  font-size: 12px;
  line-height: 1.5;
}

.explorer-header__actions {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}

.runtime-state {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  min-height: 30px;
  padding: 0 9px;
  border: 1px solid var(--border, #d9e0e8);
  border-radius: 6px;
  background: var(--surface-2, #f8fafc);
  color: var(--text-muted, #647184);
  font-size: 10px;
  white-space: nowrap;
}

.runtime-state__dot,
.provenance-badge__dot {
  display: inline-block;
  width: 7px;
  height: 7px;
  flex: 0 0 auto;
  border-radius: 50%;
  background: var(--text-muted, #647184);
}

.runtime-state--online {
  border-color: #b9dfd3;
  background: var(--success-soft, #e7f6f1);
  color: var(--success, #13795f);
}

.runtime-state--online .runtime-state__dot {
  background: var(--success, #13795f);
}

.runtime-state--offline {
  border-color: #efc2c7;
  background: var(--danger-soft, #fff0f1);
  color: var(--danger, #bf3f4d);
}

.runtime-state--offline .runtime-state__dot {
  background: var(--danger, #bf3f4d);
}

.runtime-state--checking .runtime-state__dot {
  animation: model-explorer-pulse 1.15s ease-in-out infinite;
}

.society-link {
  display: inline-flex;
  min-height: 30px;
  align-items: center;
  gap: 6px;
  padding: 0 10px;
  border: 1px solid var(--border, #d9e0e8);
  border-radius: 6px;
  background: var(--surface, #ffffff);
  color: var(--text, #182230);
  font-size: 11px;
  font-weight: 650;
  text-decoration: none;
  white-space: nowrap;
}

.society-link:hover {
  border-color: var(--primary, #2563eb);
  color: var(--primary, #2563eb);
  text-decoration: underline;
  text-underline-offset: 3px;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  margin: -1px;
  padding: 0;
  border: 0;
  clip: rect(0 0 0 0);
  white-space: nowrap;
}

.panel {
  border: 1px solid var(--border, #d9e0e8);
  border-radius: 8px;
  background: var(--surface, #ffffff);
  box-shadow: var(--shadow-sm, 0 1px 2px rgba(24, 34, 48, 0.06));
}

.catalog-panel,
.prompt-panel {
  margin-bottom: 14px;
}

.catalog-panel,
.prompt-panel,
.architecture-panel,
.attention-panel,
.analysis-panel,
.component-panel {
  min-width: 0;
  padding: 14px;
}

.panel-heading {
  justify-content: space-between;
  gap: 12px;
  min-height: 30px;
  margin-bottom: 12px;
}

.panel-heading--compact {
  min-height: 26px;
  margin-bottom: 10px;
}

.panel-heading h2 {
  margin: 3px 0 0;
  color: var(--text, #182230);
  font-size: 15px;
  font-weight: 700;
  letter-spacing: -0.015em;
  line-height: 1.25;
}

.panel-heading__actions {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 7px;
}

.provenance-cluster {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 4px;
}

.provenance-badge {
  display: inline-flex;
  min-height: 23px;
  align-items: center;
  gap: 5px;
  padding: 0 7px;
  border: 1px solid var(--border, #d9e0e8);
  border-radius: 999px;
  background: var(--surface-2, #f8fafc);
  color: var(--text-muted, #647184);
  font-size: 9px;
  font-weight: 650;
  line-height: 1;
  text-transform: lowercase;
  white-space: nowrap;
}

.provenance-badge--compact {
  min-height: 20px;
  padding-inline: 6px;
  font-size: 8px;
}

.provenance-badge--live {
  border-color: #b9dfd3;
  background: var(--success-soft, #e7f6f1);
  color: var(--success, #13795f);
}

.provenance-badge--live .provenance-badge__dot {
  background: var(--success, #13795f);
}

.provenance-badge--seeded {
  border-color: #ecd79c;
  background: var(--warning-soft, #fff6dc);
  color: var(--warning, #946200);
}

.provenance-badge--seeded .provenance-badge__dot {
  background: var(--warning, #946200);
}

.provenance-badge--reference {
  border-color: #c9d8f5;
  background: var(--accent-soft, #eaf1ff);
  color: var(--primary-focus, #1d4ed8);
}

.provenance-badge--reference .provenance-badge__dot {
  background: var(--primary, #2563eb);
}

.provenance-badge--unavailable {
  border-color: var(--border, #d9e0e8);
  background: var(--surface-2, #f8fafc);
  color: var(--text-muted, #647184);
}

.button {
  display: inline-flex;
  min-height: 32px;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 0 10px;
  border: 1px solid var(--border, #d9e0e8);
  border-radius: 6px;
  background: var(--surface, #ffffff);
  color: var(--text, #182230);
  cursor: pointer;
  font: 650 11px/1.2 var(--font, "IBM Plex Sans", "Segoe UI", sans-serif);
  white-space: nowrap;
}

.button:hover:not(:disabled) {
  border-color: var(--border-light, #c8d1dc);
  background: var(--surface-2, #f8fafc);
}

.button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.button--primary {
  border-color: var(--primary, #2563eb);
  background: var(--primary, #2563eb);
  color: #ffffff;
}

.button--primary:hover:not(:disabled) {
  border-color: var(--primary-focus, #1d4ed8);
  background: var(--primary-focus, #1d4ed8);
}

.button--secondary {
  background: var(--surface-2, #f8fafc);
}

.button--quiet {
  min-height: 28px;
  padding-inline: 8px;
  color: var(--text-muted, #647184);
  font-size: 10px;
}

.state-banner {
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 12px;
  padding: 9px 11px;
  border: 1px solid var(--border, #d9e0e8);
  border-radius: 7px;
  font-size: 11px;
  line-height: 1.45;
}

.state-banner strong {
  flex: 0 0 auto;
  font-weight: 700;
}

.state-banner--offline {
  border-color: #efc2c7;
  background: var(--danger-soft, #fff0f1);
  color: var(--danger, #bf3f4d);
}

.state-banner--error {
  border-color: #ecd79c;
  background: var(--warning-soft, #fff6dc);
  color: var(--warning, #946200);
}

.loading-state,
.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-height: 92px;
  color: var(--text-muted, #647184);
  font-size: 11px;
  line-height: 1.45;
  text-align: center;
}

.loading-state--result {
  min-height: 118px;
}

.loading-state--side {
  min-height: 128px;
  flex-direction: column;
}

.empty-state {
  flex-direction: column;
  padding: 20px 14px;
}

.empty-state strong {
  color: var(--text, #182230);
  font-size: 12px;
  font-weight: 700;
}

.empty-state--compact {
  min-height: 100px;
}

.empty-state--inline {
  min-height: 44px;
  padding: 8px 0;
}

.empty-state--result {
  min-height: 142px;
}

.empty-state__mark {
  display: grid;
  width: 28px;
  height: 28px;
  place-items: center;
  border: 1px solid var(--border, #d9e0e8);
  border-radius: 50%;
  color: var(--primary, #2563eb);
  font: 700 15px/1 var(--font-mono, monospace);
}

.spinner {
  display: inline-block;
  width: 14px;
  height: 14px;
  flex: 0 0 auto;
  border: 1.5px solid var(--border, #d9e0e8);
  border-top-color: var(--primary, #2563eb);
  border-radius: 50%;
  animation: model-explorer-spin 0.75s linear infinite;
}

.spinner--small {
  width: 11px;
  height: 11px;
  border-width: 1px;
}

.catalog-content {
  display: grid;
  gap: 12px;
}

.catalog-meta {
  justify-content: space-between;
  gap: 10px;
  color: var(--text-muted, #647184);
  font-size: 10px;
  line-height: 1.4;
}

.catalog-meta code {
  color: var(--text-dim, #354256);
}

.model-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(145px, 1fr));
  gap: 6px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.model-option {
  display: flex;
  min-width: 0;
  min-height: 38px;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 0 9px;
  border: 1px solid var(--border, #d9e0e8);
  border-radius: 6px;
  background: var(--surface, #ffffff);
  color: var(--text, #182230);
  cursor: pointer;
  text-align: left;
}

.model-option:hover {
  border-color: var(--border-light, #c8d1dc);
  background: var(--surface-2, #f8fafc);
}

.model-option--selected {
  border-color: var(--primary, #2563eb);
  background: var(--accent-soft, #eaf1ff);
  box-shadow: inset 3px 0 0 var(--primary, #2563eb);
}

.model-option__name {
  min-width: 0;
  overflow: hidden;
  font-size: 11px;
  font-weight: 650;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-option__state {
  flex: 0 0 auto;
  color: var(--text-muted, #647184);
  font-size: 8px;
  text-transform: lowercase;
}

.model-load-controls {
  flex-wrap: wrap;
  gap: 8px;
}

.field-group,
.selector-block {
  display: grid;
  gap: 5px;
}

.field-group--model {
  min-width: 190px;
  flex: 1 1 240px;
}

.field-group--endpoint {
  min-width: 155px;
  flex: 0 1 190px;
}

.field-group label,
.selector-block > label,
.selector-label-row label {
  color: var(--text-muted, #647184);
  font-size: 9px;
  font-weight: 650;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.control {
  width: 100%;
  min-height: 34px;
  padding: 0 9px;
  border: 1px solid var(--border, #d9e0e8);
  border-radius: 6px;
  background: var(--surface, #ffffff);
  color: var(--text, #182230);
  font: 12px/1.3 var(--font, "IBM Plex Sans", "Segoe UI", sans-serif);
}

.control:hover:not(:disabled) {
  border-color: var(--border-light, #c8d1dc);
}

.control:focus {
  border-color: var(--primary, #2563eb);
  outline: none;
  box-shadow: 0 0 0 2px var(--accent-soft, #eaf1ff);
}

.control:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.loaded-model {
  min-height: 32px;
  gap: 6px;
  color: var(--text-muted, #647184);
  font-size: 10px;
}

.loaded-model strong {
  color: var(--text, #182230);
  font-weight: 700;
}

.loaded-model--empty {
  font-style: italic;
}

.field-help {
  margin: 0;
  color: var(--text-muted, #647184);
  font-size: 9px;
  line-height: 1.45;
}

.auto-prompt-toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin: 0;
  color: var(--text-dim, #354256);
  font-size: 11px;
  cursor: pointer;
}

.auto-prompt-toggle input {
  width: 14px;
  height: 14px;
  accent-color: var(--primary, #2563eb);
}

.provenance-note {
  margin: 0;
  color: var(--text-muted, #647184);
  font: 9px/1.45 var(--font-mono, "IBM Plex Mono", monospace);
}

.provenance-note--inline {
  margin-left: auto;
  text-align: right;
}

.field-help--inline {
  white-space: nowrap;
}

.inline-error {
  margin: 8px 0 0;
  color: var(--danger, #bf3f4d);
  font-size: 10px;
  line-height: 1.45;
}

.prompt-form {
  display: grid;
  gap: 8px;
}

.prompt-label-row,
.selector-label-row,
.subheading-row,
.result-meta,
.next-token-card,
.attention-meta {
  justify-content: space-between;
  gap: 10px;
}

.prompt-label-row label,
.selector-label-row label {
  color: var(--text, #182230);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: none;
}

.keyboard-hint {
  color: var(--text-muted, #647184);
  font-size: 9px;
}

.control--textarea {
  min-height: 82px;
  padding: 9px;
  resize: vertical;
  font-family: var(--font-mono, "IBM Plex Mono", "Cascadia Code", monospace);
  font-size: 12px;
  line-height: 1.5;
}

.prompt-controls {
  display: flex;
  align-items: end;
  flex-wrap: wrap;
  gap: 7px;
}

.prompt-run {
  min-width: 104px;
}

.result-content,
.activation-content,
.attention-content,
.detail-content,
.architecture-content {
  display: grid;
  gap: 12px;
}

.prompt-result {
  min-height: 142px;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border, #d9e0e8);
}

.result-meta {
  color: var(--text-muted, #647184);
  font-size: 10px;
}

.subheading {
  color: var(--text-muted, #647184);
  font-size: 9px;
  font-weight: 650;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.token-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin: 6px 0 0;
  padding: 0;
  list-style: none;
}

.token-chip {
  display: inline-flex;
  max-width: 100%;
  align-items: center;
  gap: 5px;
  padding: 4px 6px;
  border: 1px solid var(--border, #d9e0e8);
  border-radius: 5px;
  background: var(--surface-2, #f8fafc);
  color: var(--text, #182230);
  font: 11px/1.2 var(--font-mono, "IBM Plex Mono", monospace);
  white-space: pre-wrap;
  word-break: break-word;
}

.token-chip__index {
  color: var(--text-muted, #647184);
  font-size: 8px;
}

.next-token-card {
  padding: 10px 12px;
  border: 1px solid #c9d8f5;
  border-radius: 7px;
  background: var(--accent-soft, #eaf1ff);
}

.next-token-card > div {
  display: grid;
  gap: 4px;
  min-width: 0;
}

.next-token-card strong {
  overflow: hidden;
  color: var(--primary-focus, #1d4ed8);
  font: 700 18px/1.1 var(--font-mono, "IBM Plex Mono", monospace);
  text-overflow: ellipsis;
  white-space: pre-wrap;
}

.next-token-card__source {
  color: var(--primary-focus, #1d4ed8);
  font: 9px/1.3 var(--font-mono, monospace);
  text-align: right;
}

.prediction-section {
  display: grid;
  gap: 5px;
}

.prediction-list {
  display: grid;
  gap: 3px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.prediction-row {
  display: grid;
  grid-template-columns: 24px minmax(0, 1fr) auto auto;
  align-items: center;
  gap: 8px;
  min-height: 28px;
  padding: 0 7px;
  border-bottom: 1px solid var(--border, #d9e0e8);
}

.prediction-row:last-child {
  border-bottom: 0;
}

.prediction-row__rank,
.prediction-row__value {
  color: var(--text-muted, #647184);
  font: 9px/1.2 var(--font-mono, monospace);
}

.prediction-row code,
.detail-title,
.data-table code,
.weight-row code {
  color: var(--text, #182230);
  font-family: var(--font-mono, "IBM Plex Mono", monospace);
}

.prediction-row code {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: pre-wrap;
}

.workspace-grid {
  display: grid;
  grid-template-columns: minmax(190px, 0.72fr) minmax(320px, 1.7fr) minmax(245px, 0.95fr);
  align-items: start;
  gap: 14px;
}

.inspector-column {
  display: grid;
  min-width: 0;
  gap: 14px;
}

.architecture-panel,
.component-panel {
  position: sticky;
  top: 0;
}

.architecture-content {
  gap: 14px;
}

.metadata-list,
.shape-list {
  display: grid;
  gap: 0;
  margin: 0;
  border-top: 1px solid var(--border, #d9e0e8);
}

.metadata-row,
.shape-row {
  justify-content: space-between;
  gap: 10px;
  min-height: 29px;
  border-bottom: 1px solid var(--border, #d9e0e8);
}

.metadata-row dt,
.shape-row dt {
  color: var(--text-muted, #647184);
  font-size: 10px;
}

.metadata-row dd,
.shape-row dd {
  max-width: 62%;
  margin: 0;
  overflow: hidden;
  color: var(--text, #182230);
  font: 10px/1.3 var(--font-mono, monospace);
  text-align: right;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.selector-block {
  gap: 6px;
}

.neuron-input-row {
  gap: 6px;
}

.neuron-input-row .control {
  min-width: 0;
}

.neuron-input-row .button {
  flex: 0 0 auto;
}

.neuron-shortcuts {
  display: grid;
  gap: 6px;
}

.neuron-chip-list {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.neuron-chip {
  min-width: 35px;
  min-height: 25px;
  padding: 0 6px;
  border: 1px solid var(--border, #d9e0e8);
  border-radius: 5px;
  background: var(--surface-2, #f8fafc);
  color: var(--text-dim, #354256);
  cursor: pointer;
  font: 10px/1 var(--font-mono, monospace);
}

.neuron-chip:hover,
.neuron-chip--selected {
  border-color: var(--primary, #2563eb);
  background: var(--accent-soft, #eaf1ff);
  color: var(--primary-focus, #1d4ed8);
}

.attention-content {
  gap: 9px;
}

.attention-meta {
  color: var(--text-muted, #647184);
  font: 9px/1.3 var(--font-mono, monospace);
}

.heatmap-wrap {
  max-width: 100%;
  overflow: auto;
  padding: 8px;
  border: 1px solid var(--border, #d9e0e8);
  border-radius: 6px;
  background: var(--surface-2, #f8fafc);
}

.heatmap-grid {
  display: grid;
  min-width: 260px;
  gap: 2px;
}

.heatmap-cell {
  display: block;
  min-width: 8px;
  min-height: 16px;
  border-radius: 2px;
}

.heatmap-labels {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  color: var(--text-muted, #647184);
  font: 9px/1.2 var(--font-mono, monospace);
}

.heatmap-labels span {
  max-width: 90px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: pre;
}

.metadata-list--compact {
  margin-top: 2px;
}

.analysis-actions {
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 12px;
}

.analysis-actions .button {
  min-height: 29px;
  font-size: 10px;
}

.shape-list {
  grid-template-columns: repeat(3, minmax(0, 1fr));
  border: 1px solid var(--border, #d9e0e8);
  border-radius: 6px;
  overflow: hidden;
}

.shape-row {
  display: grid;
  align-content: center;
  gap: 3px;
  min-height: 48px;
  padding: 6px 8px;
  border-right: 1px solid var(--border, #d9e0e8);
  border-bottom: 1px solid var(--border, #d9e0e8);
}

.shape-row:nth-child(3n) {
  border-right: 0;
}

.shape-row dt,
.shape-row dd {
  max-width: none;
  text-align: left;
}

.shape-row dd {
  color: var(--text, #182230);
  font-size: 10px;
}

.activation-table-wrap {
  overflow-x: auto;
}

.data-table {
  width: 100%;
  min-width: 300px;
  border-collapse: collapse;
  font-size: 10px;
}

.data-table th,
.data-table td {
  padding: 6px 7px;
  border-bottom: 1px solid var(--border, #d9e0e8);
  text-align: left;
  white-space: nowrap;
}

.data-table thead th {
  color: var(--text-muted, #647184);
  font: 9px/1.2 var(--font-mono, monospace);
  text-transform: uppercase;
}

.data-table tbody th {
  max-width: 160px;
  overflow: hidden;
  color: var(--text, #182230);
  font: 10px/1.2 var(--font-mono, monospace);
  text-overflow: ellipsis;
}

.data-table td {
  color: var(--text-dim, #354256);
  font: 10px/1.2 var(--font-mono, monospace);
}

.lens-section {
  display: grid;
  gap: 7px;
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid var(--border, #d9e0e8);
}

.lens-list {
  display: grid;
  gap: 3px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.lens-row {
  display: grid;
  grid-template-columns: 38px minmax(0, 1fr) minmax(0, 1.4fr);
  align-items: center;
  gap: 8px;
  min-height: 29px;
  padding: 0 6px;
  border-bottom: 1px solid var(--border, #d9e0e8);
}

.lens-row__layer {
  min-height: 24px;
  padding: 0 5px;
  border: 1px solid var(--border, #d9e0e8);
  border-radius: 4px;
  background: var(--surface-2, #f8fafc);
  color: var(--primary-focus, #1d4ed8);
  cursor: pointer;
  font: 10px/1 var(--font-mono, monospace);
}

.lens-row__layer:hover {
  border-color: var(--primary, #2563eb);
}

.lens-row__token {
  overflow: hidden;
  color: var(--text, #182230);
  font: 10px/1.2 var(--font-mono, monospace);
  text-overflow: ellipsis;
  white-space: pre-wrap;
}

.lens-row__tokens {
  overflow: hidden;
  color: var(--text-muted, #647184);
  font-size: 9px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.inspector-tabs {
  display: flex;
  gap: 2px;
  margin-bottom: 10px;
  padding-bottom: 5px;
  border-bottom: 1px solid var(--border, #d9e0e8);
}

.inspector-tab {
  min-height: 28px;
  padding: 0 8px;
  border: 1px solid transparent;
  border-radius: 5px;
  background: transparent;
  color: var(--text-muted, #647184);
  cursor: pointer;
  font-size: 10px;
  font-weight: 650;
}

.inspector-tab:hover {
  border-color: var(--border, #d9e0e8);
  background: var(--surface-2, #f8fafc);
}

.inspector-tab[aria-selected="true"] {
  border-color: #c9d8f5;
  background: var(--accent-soft, #eaf1ff);
  color: var(--primary-focus, #1d4ed8);
}

.component-detail {
  min-width: 0;
}

.detail-title {
  color: var(--text, #182230);
  font-size: 14px;
  font-weight: 700;
  overflow-wrap: anywhere;
}

.detail-description {
  margin: 4px 0 0;
  color: var(--text-muted, #647184);
  font-size: 10px;
  line-height: 1.5;
}

.detail-list,
.weight-list {
  display: grid;
  gap: 5px;
}

.detail-list__row {
  display: grid;
  grid-template-columns: 38px minmax(0, 1fr) auto;
  align-items: center;
  gap: 6px;
  min-height: 26px;
  padding: 0 5px;
  border: 0;
  border-bottom: 1px solid var(--border, #d9e0e8);
  background: transparent;
  color: var(--text-dim, #354256);
  cursor: pointer;
  font: 10px/1.2 var(--font-mono, monospace);
  text-align: left;
}

.detail-list__row:hover {
  background: var(--surface-2, #f8fafc);
}

.detail-list__row span:nth-child(2) {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.detail-list__row span:last-child {
  color: var(--text-muted, #647184);
}

.neuron-token-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 6px;
  align-items: center;
  min-height: 25px;
  border-bottom: 1px solid var(--border, #d9e0e8);
  color: var(--text-dim, #354256);
  font: 10px/1.2 var(--font-mono, monospace);
}

.neuron-token-row code {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: pre-wrap;
}

.neuron-token-row .field-help {
  grid-column: 1 / -1;
  margin-top: -4px;
  padding-left: 2px;
  font-size: 8px;
}

.weight-row {
  display: grid;
  grid-template-columns: 32px minmax(0, 1fr) 58px;
  align-items: center;
  gap: 6px;
  min-height: 23px;
  color: var(--text-muted, #647184);
  font: 9px/1.2 var(--font-mono, monospace);
}

.weight-track {
  display: block;
  height: 5px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--border, #d9e0e8);
}

.weight-track span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--primary, #2563eb);
}

.weight-row code {
  overflow: hidden;
  font-size: 9px;
  text-align: right;
  text-overflow: ellipsis;
}

@keyframes model-explorer-spin {
  to { transform: rotate(360deg); }
}

@keyframes model-explorer-pulse {
  0%, 100% { opacity: 0.45; }
  50% { opacity: 1; }
}

@media (max-width: 1120px) {
  .workspace-grid {
    grid-template-columns: minmax(190px, 0.75fr) minmax(0, 1.7fr);
  }

  .component-panel {
    grid-column: 1 / -1;
    position: static;
  }

  .component-detail {
    min-height: 120px;
  }
}

@media (max-width: 760px) {
  .model-explorer__frame {
    padding: 12px;
  }

  .explorer-header {
    align-items: flex-start;
    flex-direction: column;
    gap: 10px;
  }

  .explorer-header__actions {
    width: 100%;
    justify-content: space-between;
  }

  .catalog-meta,
  .result-meta {
    align-items: flex-start;
    flex-direction: column;
  }

  .workspace-grid {
    grid-template-columns: 1fr;
  }

  .architecture-panel,
  .component-panel {
    position: static;
  }

  .component-panel {
    grid-column: auto;
  }

  .model-list {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 480px) {
  .model-list {
    grid-template-columns: 1fr;
  }

  .model-load-controls,
  .prompt-controls {
    align-items: stretch;
    flex-direction: column;
  }

  .field-group--model,
  .field-group--endpoint,
  .prompt-run,
  .model-load-controls > .button,
  .prompt-controls > .button {
    width: 100%;
  }

  .shape-list {
    grid-template-columns: 1fr 1fr;
  }

  .shape-row:nth-child(3n) {
    border-right: 1px solid var(--border, #d9e0e8);
  }

  .shape-row:nth-child(2n) {
    border-right: 0;
  }

  .lens-row {
    grid-template-columns: 38px minmax(0, 1fr);
  }

  .lens-row__tokens {
    grid-column: 2;
  }
}

@media (prefers-reduced-motion: reduce) {
  .spinner,
  .runtime-state--checking .runtime-state__dot {
    animation: none;
  }
}
</style>
