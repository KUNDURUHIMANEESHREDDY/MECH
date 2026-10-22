<template>
  <div
    id="neural-explorer-root"
    class="flex flex-col h-full bg-[var(--bg)] text-[var(--ink)] font-['Inter',system-ui,sans-serif] overflow-hidden"
  >
    <!-- Title. The shell's navigation contract is that every tool view names
    itself: the smoke test asserts a heading is present in the view it opens.
    This view had none, so the assertion had nothing to hold. -->
    <div class="flex items-baseline gap-2 px-4 pt-2.5 shrink-0">
      <h2 class="text-[var(--ink)] text-sm font-semibold m-0">Neural Explorer</h2>
      <span class="text-[10px] text-[var(--ink-muted-48)] uppercase tracking-widest">Activations</span>
    </div>

    <!-- Prompt bar -->
    <div class="flex items-center gap-2.5 px-4 py-2.5 bg-[var(--surface-pearl)] border-b border-[var(--border)] shrink-0">
      <span class="text-[10px] text-[var(--ink-muted-48)] uppercase tracking-widest">Prompt</span>
      <input
        id="ne-prompt-input"
        aria-label="Prompt"
        v-model="prompt"
        @keydown.enter="handleRunPrompt"
        placeholder="Run a prompt to populate activations…"
        class="flex-1 min-w-0 bg-[var(--bg)] border border-[var(--border)] text-[var(--ink)] rounded-md px-2.5 py-1.5 text-xs"
      />
      <button
        id="ne-run-prompt-btn"
        @click="handleRunPrompt"
        :disabled="promptRunning || !selectedModel"
        class="rounded-lg px-4 py-1.5 text-xs font-semibold text-white cursor-pointer disabled:cursor-default disabled:opacity-65"
        :class="promptRunning ? 'bg-[var(--primary)]' : 'bg-[var(--primary)]'"
      >
        {{ promptRunning ? 'Running…' : 'Run Prompt' }}
      </button>
      <button
        id="ne-refresh-btn"
        @click="loadTree"
        title="Reload layer data"
        class="bg-transparent border border-[var(--border)] rounded-lg px-3.5 py-1.5 text-xs text-[var(--primary-focus)] cursor-pointer"
      >
        Refresh
      </button>
    </div>

    <!-- Status message -->
    <div
      v-if="promptMsg"
      class="px-4 py-1.5 text-[11px] shrink-0 border-b border-[var(--border)]"
      :class="promptMsg.ok ? 'text-[var(--primary)] bg-[var(--success)]/[0.09]' : 'text-[var(--danger)] bg-[var(--danger)]/[0.09]'"
    >
      {{ promptMsg.text }}
    </div>

    <!-- Availability. The backend's own reason, shown verbatim: a surface
         that could not read anything says so, instead of rendering an empty
         result that reads like a measurement. -->
    <div
      v-if="modelError || treeError"
      class="px-4 py-1.5 text-[11px] shrink-0 border-b border-[var(--border)] text-[var(--danger)] bg-[var(--danger)]/[0.09]"
      role="alert"
      data-testid="ne-unavailable"
    >
      <span v-if="modelError">{{ modelError }}</span>
      <span v-else>{{ treeError }}</span>
    </div>

    <!-- Token strip -->
    <div
      v-if="strTokens.length > 0"
      data-testid="ne-token-strip"
      class="flex flex-wrap gap-1 px-4 py-1.5 shrink-0 bg-[var(--surface-pearl)] border-b border-[var(--border)]"
    >
      <span
        v-for="(t, i) in strTokens"
        :key="i"
        :data-testid="`ne-token-chip-${i}`"
        class="text-[10px] px-1.5 py-0.5 rounded"
        :class="i === selTokIdx
          ? 'bg-[var(--primary)]/[0.2] border border-[var(--primary)] text-[var(--primary-focus)] font-semibold'
          : 'bg-[var(--bg)] border border-[var(--border)] text-[var(--ink-muted-48)]'"
      >
        {{ fmtToken(t) }}
      </span>
    </div>

    <!-- Main content -->
    <div class="flex flex-1 min-h-0">
      <!-- Model tree sidebar -->
      <div class="w-[220px] min-w-[180px] bg-[var(--surface-pearl)] border-r border-[var(--border)] overflow-y-auto py-3 flex flex-col">
        <div class="px-3 pb-3 border-b border-[var(--border)]">
          <div class="text-[10px] text-[var(--ink-muted-48)] mb-1.5 uppercase tracking-widest">Model</div>
          <select
            id="neural-explorer-model-select"
            aria-label="Model"
            :value="selectedModel?.id || ''"
            @change="onModelChange"
            class="w-full bg-[var(--bg)] border border-[var(--border)] text-[var(--ink)] rounded-md px-2 py-1.5 text-xs"
          >
            <option v-for="m in models" :key="m.id" :value="m.id">{{ m.label }}</option>
          </select>
        </div>

        <div class="flex-1 overflow-y-auto">
          <div v-for="layer in treeData?.layers" :key="layer.layer_index">
            <button
              :id="`layer-btn-${layer.layer_index}`"
              @click="() => { selectLayer(layer); expandedLayers.toggle(layer.layer_index); }"
              class="w-full text-left py-[7px] px-3.5 border-none text-xs cursor-pointer flex items-center gap-1.5"
              :class="selectedLayer?.layer_index === layer.layer_index
                ? 'bg-[var(--surface-pearl)] border-l-[3px] border-l-[var(--primary)] text-[var(--primary-focus)]'
                : 'bg-transparent border-l-[3px] border-l-transparent text-[var(--primary-focus)]'"
            >
              <span class="inline-flex text-[var(--ink-muted-48)]">
                <svg
                  width="8" height="8" viewBox="0 0 8 8"
                  class="transition-transform duration-150"
                  :style="{ transform: expandedLayers.has(layer.layer_index) ? 'rotate(90deg)' : 'none' }"
                >
                  <polygon points="2,0 8,4 2,8" fill="currentColor" />
                </svg>
              </span>
              <span>Layer {{ layer.layer_index }}</span>
            </button>

            <div v-if="expandedLayers.has(layer.layer_index)" class="pl-6">
              <button
                v-for="h in layer.attention_heads_preview"
                :key="h.head_index"
                :id="`tree-head-btn-${layer.layer_index}-${h.head_index}`"
                :title="`Head ${h.head_index} — Q L2: ${metric(h.q_weight_l2, 3)}`"
                @click="() => { selectLayer(layer); handleSelectHead(layer.layer_index, h.head_index); }"
                class="block w-full text-left py-[3px] px-2 bg-transparent border-none text-[11px] cursor-pointer"
                :class="selectedHead?.layer === layer.layer_index && selectedHead?.head_index === h.head_index
                  ? 'text-[var(--primary)]' : 'text-[var(--ink-muted-48)]'"
              >
                H{{ h.head_index }}
                <span v-if="h.top_token" class="text-[var(--primary)] text-[10px] ml-1">· {{ fmtToken(h.top_token) }}</span>
              </button>
              <div class="text-[10px] text-[var(--ink-muted-48)] pl-2 mt-1">
                {{ metric(layer.num_mlp_neurons, 0) }} MLP neurons
              </div>
              <template v-if="layer.mlp_neurons_preview?.length">
                <button
                  v-for="n in layer.mlp_neurons_preview.slice(0, 16)"
                  :key="n.neuron_index"
                  :id="`neuron-btn-${layer.layer_index}-${n.neuron_index}`"
                  @click="handleSelectNeuron({ layer: layer.layer_index, neuron_index: n.neuron_index })"
                  class="block w-full text-left py-[3px] px-2 bg-transparent border-none text-[11px] cursor-pointer"
                  :class="selectedNeuron?.layer === layer.layer_index && selectedNeuron?.neuron_index === n.neuron_index
                    ? 'text-[var(--primary)]' : 'text-[var(--ink-muted-48)]'"
                >
                  N{{ n.neuron_index }}
                  <span v-if="n.top_token" class="text-[var(--primary)] text-[10px] ml-1">· {{ fmtToken(n.top_token) }}</span>
                </button>
              </template>
              <div v-else class="text-[10px] text-[var(--ink-muted-48)] pl-2">Run a prompt to see active neurons</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Layer detail -->
      <div v-if="selectedLayer" class="w-[260px] bg-[var(--surface-pearl)] border-r border-[var(--border)] overflow-y-auto p-4">
        <div class="text-[13px] font-bold text-[var(--primary)] mb-3">
          Layer {{ selectedLayer.layer_index }}
          <span class="text-[10px] text-[var(--ink-muted-48)] font-normal ml-2">d={{ metric(selectedLayer.residual_stream_dim, 0) }}</span>
        </div>
        <div class="text-[10px] text-[var(--ink-muted-48)] uppercase tracking-widest mb-1.5">
          Attention Heads ({{ metric(selectedLayer.num_attention_heads, 0) }}) · click to inspect
        </div>
        <div class="grid grid-cols-4 gap-1 mb-4">
          <button
            v-for="h in selectedLayer.attention_heads_preview"
            :key="h.head_index"
            :id="`detail-head-btn-${selectedLayer.layer_index}-${h.head_index}`"
            @click="handleSelectHead(selectedLayer.layer_index, h.head_index)"
            class="py-1 px-0.5 rounded text-center text-[10px] cursor-pointer"
            :class="selectedHead?.layer === selectedLayer.layer_index && selectedHead?.head_index === h.head_index
              ? 'bg-[var(--bg)] border border-[var(--primary-focus)]' : 'bg-[var(--bg)] border border-[var(--border)] text-[var(--ink-muted-48)]'"
          >
            H{{ h.head_index }}
            <div v-if="h.top_token" class="text-[9px] text-[var(--primary)] mt-0.5">{{ fmtToken(h.top_token) }}</div>
          </button>
        </div>

        <div class="text-[10px] text-[var(--ink-muted-48)] uppercase tracking-widest mb-1.5">Top Active Neurons</div>
        <template v-if="selectedLayer.mlp_neurons_preview?.length">
          <div class="grid grid-cols-4 gap-[3px]">
            <div
              v-for="n in selectedLayer.mlp_neurons_preview?.slice(0, 32)"
              :key="n.neuron_index"
              :title="n.label"
              class="py-1 px-0.5 rounded text-center text-[10px] bg-[var(--bg)] border border-[var(--border)] text-[var(--ink-muted-48)] cursor-default"
            >
              {{ n.neuron_index }}
              <div v-if="n.top_token" class="text-[9px] text-[var(--primary)] mt-0.5">{{ fmtToken(n.top_token) }}</div>
            </div>
          </div>
        </template>
        <div v-else class="text-[11px] text-[var(--ink-muted-48)]">Run a prompt to populate top active neurons.</div>
      </div>

      <!-- Neuron map (UMAP) -->
      <div v-if="selectedLayer?.mlp_neurons_preview" class="w-[430px] bg-[var(--surface-pearl)] border-r border-[var(--border)] overflow-y-auto p-4 shrink-0">
        <div class="text-[13px] font-bold text-[var(--primary)] mb-3">
          Neuron Map — Layer {{ selectedLayer.layer_index }}
        </div>
        <div v-if="!hasActivations" class="text-[11px] text-[var(--ink-muted-48)] mb-2.5">
          No activations yet — run a prompt above to populate the map.
        </div>
        <NeuronUMAPWrapper
          :points="buildLayerNeuronPoints(selectedLayer.mlp_neurons_preview, selectedLayer.layer_index)"
          :selectedId="selectedNeuron && selectedNeuron.layer === selectedLayer.layer_index
            ? idForLayerNeuron(selectedNeuron.layer, selectedNeuron.neuron_index)
            : null"
          @selectNeuron="(id) => {
            const parsed = parseNeuronId(id);
            handleSelectNeuron(parsed ? { layer: parsed.layer, neuron_index: parsed.neuron } : null);
          }"
          :height="460"
        />
      </div>

      <!-- Right panel: loading / head detail / neuron detail -->
      <div v-if="loading" class="flex-1 flex items-center justify-center text-[var(--ink-muted-48)] text-[14px]">
        Loading model tree…
      </div>
      <div v-else-if="selectedHead" class="flex-1 overflow-y-auto p-5 bg-[var(--bg)]">
        <div class="text-[18px] font-bold text-[var(--primary-focus)] mb-1">
          GPT-2 · L{{ selectedHead.layer }}H{{ selectedHead.head_index }}
        </div>
        <div class="text-[11px] text-[var(--ink-muted-48)] mb-4">
          Attention pattern — rows = query tokens, cols = key tokens
        </div>
        <div v-if="headLoading" class="text-[var(--ink-muted-48)] text-[13px]">Loading attention pattern…</div>
        <div
          v-if="headError"
          class="bg-[var(--danger)]/[0.09] border border-[var(--danger)]/[0.33] text-[var(--danger)] rounded-lg px-4 py-3 text-xs mb-4 inline-block"
        >
          {{ headError }}
        </div>
        <div v-if="headDetail" class="bg-[var(--bg-elev)] rounded-[10px] p-4 border border-[var(--border)] inline-block">
          <AttentionHeatmap
            :matrix="headDetail.matrix"
            :tokens="headDetail.str_tokens || []"
            :hoveredToken="hoveredToken"
            @hoverToken="hoveredToken = $event"
          />
        </div>
      </div>
      <div v-else class="flex-1 overflow-y-auto p-5 bg-[var(--bg)]">
        <div v-if="neuronError" class="flex-1 p-5" role="alert" data-testid="ne-neuron-error">
          <div class="bg-[var(--danger)]/[0.09] border border-[var(--danger)]/[0.33] text-[var(--danger)] rounded-lg px-4 py-3 text-xs">
            {{ neuronError }}
          </div>
        </div>
        <div v-else-if="!neuronDetail" class="flex-1 flex items-center justify-center text-[var(--ink-muted-48)] text-[14px]">
          Select a neuron to inspect
        </div>
        <template v-else>
          <div class="flex items-center gap-3 mb-5">
            <div>
              <div class="text-[18px] font-bold text-[var(--primary-focus)]">
                {{ selectedModel?.label || 'GPT-2' }} · {{ neuronDetail.id || `L${neuronDetail.layer}N${neuronDetail.neuron_index}` }}
              </div>
              <div class="text-[11px] text-[var(--ink-muted-48)] mt-0.5">{{ neuronDetail.description }}</div>
              <div v-if="activatesOn" class="mt-2 flex items-center gap-2">
                <span class="text-[10px] text-[var(--ink-muted-48)] uppercase tracking-widest">Activates on</span>
                <code class="bg-[var(--primary)]/[0.13] border border-[var(--primary)]/[0.33] text-[var(--primary-focus)] rounded-md px-2.5 py-0.5 text-[14px] font-semibold">
                  {{ fmtToken(activatesOn.token) }}
                </code>
              </div>
            </div>
            <div class="ml-auto flex gap-2">
              <span class="bg-[var(--primary)]/[0.13] text-[var(--primary)] border border-[var(--primary)]/[0.33] rounded-full px-2.5 py-0.5 text-[11px] mr-1 mb-1 inline-block">
                {{ neuronDetail.component }}
              </span>
            </div>
          </div>

          <div class="grid grid-cols-2 gap-4">
            <!-- Weight Summary -->
            <div class="bg-[var(--bg-elev)] rounded-[10px] p-4 border border-[var(--border)]">
              <div class="text-[11px] text-[var(--ink-muted-48)] mb-1 uppercase tracking-widest">Weight Summary</div>
              <div class="grid grid-cols-2 gap-1.5 text-[11px]">
                <span class="text-[var(--ink-muted-48)]">Bias</span>
                <span class="text-[var(--body-muted)]">{{ metric(neuronDetail.bias) }}</span>
                <span class="text-[var(--ink-muted-48)]">In Weight L2</span>
                <span class="text-[var(--body-muted)]">{{ metric(neuronDetail.in_weight_l2) }}</span>
                <span class="text-[var(--ink-muted-48)]">Out Weight L2</span>
                <span class="text-[var(--body-muted)]">{{ metric(neuronDetail.out_weight_l2) }}</span>
                <span class="text-[var(--ink-muted-48)]">In Mean</span>
                <span class="text-[var(--body-muted)]">{{ metric(neuronDetail.in_weight_stats?.mean) }}</span>
                <span class="text-[var(--ink-muted-48)]">Out Mean</span>
                <span class="text-[var(--body-muted)]">{{ metric(neuronDetail.out_weight_stats?.mean) }}</span>
              </div>
            </div>

            <!-- Activation Stats -->
            <div class="bg-[var(--bg-elev)] rounded-[10px] p-4 border border-[var(--border)]">
              <div class="text-[11px] text-[var(--ink-muted-48)] mb-1 uppercase tracking-widest">Activation Stats</div>
              <template v-if="neuronDetail.activation_stats">
                <div class="grid grid-cols-2 gap-1.5 text-[11px]">
                  <span class="text-[var(--ink-muted-48)]">Mean</span>
                  <span class="text-[var(--body-muted)]">{{ metric(neuronDetail.activation_stats.mean) }}</span>
                  <span class="text-[var(--ink-muted-48)]">Std</span>
                  <span class="text-[var(--body-muted)]">{{ metric(neuronDetail.activation_stats.std) }}</span>
                  <span class="text-[var(--ink-muted-48)]">Min</span>
                  <span class="text-[var(--body-muted)]">{{ metric(neuronDetail.activation_stats.min) }}</span>
                  <span class="text-[var(--ink-muted-48)]">Max</span>
                  <span class="text-[var(--body-muted)]">{{ metric(neuronDetail.activation_stats.max) }}</span>
                </div>
              </template>
              <span v-else class="text-[var(--ink-muted-48)] text-[11px]">No activations — run a prompt first</span>
              <!-- Mini histogram -->
              <div v-if="neuronDetail.activation_histogram?.counts" class="flex items-end h-10 gap-0.5 my-2">
                <div
                  v-for="(count, i) in neuronDetail.activation_histogram.counts"
                  :key="i"
                  :title="`[${neuronDetail.activation_histogram.bins[i]}, ${neuronDetail.activation_histogram.bins[i+1]}): ${count}`"
                  class="flex-1 rounded-t-[2px] transition-[height] duration-300"
                  :class="i === 4 || i === 5 ? 'bg-[var(--primary)]/40' : 'bg-[var(--bg)]'"
                  :style="{ height: `${Math.max(3, (count / maxHistCount) * 100)}%` }"
                />
              </div>
            </div>

            <!-- Top Input Weights -->
            <div class="bg-[var(--bg-elev)] rounded-[10px] p-4 border border-[var(--border)]">
              <div class="text-[11px] text-[var(--ink-muted-48)] mb-2 uppercase tracking-widest">Top Input Weights</div>
              <div v-for="w in topWeightsPositive" :key="w.dim" class="flex items-center gap-1 text-[10px] mb-0.5">
                <span class="text-[var(--ink-muted-48)] min-w-[40px] text-[9px]">D{{ w.dim }}</span>
                <div class="flex-1 h-1.5 bg-[var(--border)] rounded-sm overflow-hidden">
                  <div
                    class="h-full rounded-sm"
                    :class="w.weight >= 0 ? 'bg-[var(--primary)]' : 'bg-[var(--danger)]'"
                    :style="{ width: `${Math.min(100, (Math.abs(w.weight) / maxAbsWeight) * 100)}%` }"
                  />
                </div>
                <span class="min-w-[50px] text-right" :class="w.weight >= 0 ? 'text-[var(--primary)]' : 'text-[var(--danger)]'">
                  {{ w.weight.toFixed(4) }}
                </span>
              </div>
              <div v-for="w in topWeightsNegative" :key="'neg-' + w.dim" class="flex items-center gap-1 text-[10px] mb-0.5">
                <span class="text-[var(--ink-muted-48)] min-w-[40px] text-[9px]">D{{ w.dim }}</span>
                <div class="flex-1 h-1.5 bg-[var(--border)] rounded-sm overflow-hidden">
                  <div
                    class="h-full rounded-sm bg-[var(--danger)]"
                    :style="{ width: `${Math.min(100, (Math.abs(w.weight) / maxAbsWeight) * 100)}%` }"
                  />
                </div>
                <span class="text-[var(--danger)] min-w-[50px] text-right">{{ w.weight.toFixed(4) }}</span>
              </div>
            </div>

            <!-- Per-Token Activations -->
            <div class="bg-[var(--bg-elev)] rounded-[10px] p-4 border border-[var(--border)]">
              <div class="text-[11px] text-[var(--ink-muted-48)] mb-2 uppercase tracking-widest">Per-Token Activations</div>
              <template v-if="neuronDetail.per_token_activations?.length">
                <div v-for="ta in neuronDetail.per_token_activations.slice(0, 12)" :key="ta.token_index" class="flex items-center gap-2 mb-1">
                  <code class="bg-[var(--surface-pearl)] rounded-sm px-1.5 py-[1px] text-[11px] text-[var(--primary-focus)] min-w-[80px]">
                    {{ ta.token }}
                  </code>
                  <div class="flex-1 h-1 bg-[var(--border)] rounded-sm relative overflow-visible">
                    <div
                      class="absolute h-full rounded-sm"
                      :class="ta.activation >= 0 ? 'bg-[var(--primary)]' : 'bg-[var(--danger)]'"
                      :style="{
                        left: '50%',
                        width: `${Math.min(50, Math.abs(ta.activation) / 5 * 100)}%`,
                        transform: ta.activation >= 0 ? 'translateX(0)' : 'translateX(-100%)',
                      }"
                    />
                  </div>
                  <span class="text-[10px] text-[var(--ink-muted-48)] min-w-[60px] text-right">{{ ta.activation.toFixed(4) }}</span>
                  <span v-if="ta.pre_activation !== null" class="text-[9px] text-[var(--ink-muted-48)]">pre: {{ ta.pre_activation.toFixed(3) }}</span>
                </div>
              </template>
              <span v-else class="text-[var(--ink-muted-48)] text-[11px]">No cached activations.</span>
            </div>

            <!-- Nearest Neurons -->
            <div class="bg-[var(--bg-elev)] rounded-[10px] p-4 border border-[var(--border)]">
              <div class="text-[11px] text-[var(--ink-muted-48)] mb-2 uppercase tracking-widest">Nearest Neurons (by weights)</div>
              <template v-if="neuronDetail.nearest_neurons?.length">
                <div v-for="n in neuronDetail.nearest_neurons" :key="n.neuron_index" class="flex justify-between text-[11px] mb-1">
                  <span class="text-[var(--ink-muted-48)]">L{{ n.layer }}N{{ n.neuron_index }}</span>
                  <span class="text-[var(--primary)]">{{ percent(n.similarity) }}</span>
                </div>
              </template>
              <span v-else class="text-[var(--ink-muted-48)] text-[11px]">No nearest neighbors.</span>
            </div>

            <!-- Patch Experiment -->
            <div class="bg-[var(--bg-elev)] rounded-[10px] p-4 border border-[var(--border)] col-span-2">
              <div class="text-[11px] text-[var(--ink-muted-48)] mb-2.5 uppercase tracking-widest">Activation Patch Experiment</div>
              <div class="flex items-center gap-3 flex-wrap">
                <div>
                  <label for="patch-value-input" class="text-[11px] text-[var(--ink-muted-48)] block mb-1">Patch Value</label>
                  <input
                    id="patch-value-input"
                    v-model.number="patchVal"
                    type="number" step="0.1"
                    class="w-[90px] bg-[var(--bg)] border border-[var(--border)] text-[var(--ink)] rounded-md px-2 py-1.5 text-[13px]"
                  />
                </div>
                <button
                  id="patch-run-btn"
                  @click="handlePatch"
                  :disabled="!selectedNeuron"
                  :title="selectedNeuron
                    ? 'Patch this neuron and re-read it from the live model'
                    : 'Select a neuron first: there is nothing to patch.'"
                  class="rounded-lg px-5 py-2 text-[13px] font-semibold text-white cursor-pointer mt-5 bg-[var(--primary)] disabled:cursor-default disabled:opacity-60"
                >
                  <svg width="12" height="12" viewBox="0 0 12 12" class="mr-1.5 inline-block align-middle">
                    <polygon points="2,0 12,6 2,12" fill="currentColor" />
                  </svg>
                  Run Patch
                </button>
                <p class="text-[10px] text-[var(--ink-muted-48)] mt-2 max-w-[420px]">
                  Intervention prompt: <code>The Eiffel Tower is in</code> — the same prompt is read before and
                  after the patch, so the two readings differ only by the patched neuron.
                </p>
                <div
                  v-if="patchError"
                  class="bg-[var(--danger)]/[0.09] border border-[var(--danger)]/[0.33] text-[var(--danger)] rounded-lg px-4 py-3 text-xs mt-4"
                  role="alert"
                  data-testid="ne-patch-error"
                >
                  {{ patchError }}
                </div>
                <template v-if="patchResult && patchResult.status === 'ok'">
                  <div class="flex gap-4 mt-4 flex-wrap">
                    <div class="bg-[var(--bg)] rounded-lg px-3.5 py-2 text-center">
                      <div class="text-[10px] text-[var(--ink-muted-48)]">Before</div>
                      <code class="text-[var(--primary-focus)] text-[13px]">{{ patchResult.top_token_before ?? 'not measured' }}</code>
                    </div>
                    <svg width="20" height="14" viewBox="0 0 20 14" class="text-[var(--primary)] self-center">
                      <path d="M0 7 H16 M11 2 L16 7 L11 12" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" />
                    </svg>
                    <div class="bg-[var(--bg)] rounded-lg px-3.5 py-2 text-center">
                      <div class="text-[10px] text-[var(--ink-muted-48)]">After</div>
                      <code class="text-[var(--primary)] text-[13px]">{{ patchResult.top_token_after ?? 'not measured' }}</code>
                    </div>
                    <div class="bg-[var(--bg)] rounded-lg px-3.5 py-2 text-center">
                      <div class="text-[10px] text-[var(--ink-muted-48)]">Δ logit</div>
                      <span class="text-[13px] font-bold"
                        :class="typeof patchResult.delta === 'number' && Number.isFinite(patchResult.delta)
                          ? (patchResult.delta > 0 ? 'text-[var(--primary)]' : 'text-[var(--danger)]')
                          : 'text-[var(--ink-muted-48)]'"
                      >
                        {{ metric(patchResult.delta) }}
                      </span>
                    </div>
                  </div>
                </template>
              </div>
            </div>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, watch, onMounted, onBeforeUnmount } from 'vue';
import { api } from '../services/api';
import NeuronUMAPWrapper from './visualizations/neuron-umap/NeuronUMAPWrapper.vue';
import { buildLayerNeuronPoints, idForLayerNeuron, parseNeuronId } from './visualizations/neuron-umap/data';
import AttentionHeatmap from './visualizations/vue-panels/AttentionHeatmap.vue';

/* ── helpers ────────────────────────────────────── */
function fmtToken(tok: string) {
  if (!tok) return '';
  return String(tok).replaceAll('Ġ', '␣').replaceAll('Ċ', '⏎').trim();
}

/**
 * One measured number, or the words "not measured".
 *
 * A field the backend did not return is not zero and not a rounded guess.
 * Rendering it as 0 would put a number on screen that nobody measured.
 */
function metric(value: unknown, digits = 4): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'not measured';
}

/** A measured percentage, or the words "not measured". */
function percent(value: unknown, digits = 1): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? `${(value * 100).toFixed(digits)}%`
    : 'not measured';
}

class ToggleSet {
  private s = reactive(new Set<number>());
  has(v: number) { return this.s.has(v); }
  toggle(v: number) { this.s.has(v) ? this.s.delete(v) : this.s.add(v); }
}

/* ── state ──────────────────────────────────────── */
const models = ref<any[]>([]);
const selectedModel = ref<any>(null);
const treeData = ref<any>(null);
const selectedLayer = ref<any>(null);
const selectedNeuron = ref<any>(null);
const neuronDetail = ref<any>(null);
const patchResult = ref<any>(null);
const selectedHead = ref<any>(null);
const headDetail = ref<any>(null);
const headError = ref<string | null>(null);
const headLoading = ref(false);
/**
 * Why a surface has nothing to show.
 *
 * Each holds the backend's own reason, shown verbatim. A view that cannot
 * say why it is empty must not render an empty result as if it were one.
 */
const modelError = ref<string | null>(null);
const treeError = ref<string | null>(null);
const neuronError = ref<string | null>(null);
const patchError = ref<string | null>(null);
const loading = ref(false);
const prompt = ref('The capital of France is');
const promptRunning = ref(false);
const promptMsg = ref<{ ok: boolean; text: string } | null>(null);
const hasActivations = ref(false);
const strTokens = ref<string[]>([]);
const expandedLayers = reactive(new ToggleSet());
const patchVal = ref(3.0);
const hoveredToken = ref<string | null>(null);

/* ── computed ────────────────────────────────────── */
const activatesOn = computed(() => {
  const perTok = neuronDetail.value?.per_token_activations || [];
  if (perTok.length === 0) return null;
  return perTok.reduce((best: any, ta: any) => (ta.activation > best.activation ? ta : best), perTok[0]);
});

const topWeightsPositive = computed(() => neuronDetail.value?.top_input_weights_positive?.slice(0, 8) || []);
const topWeightsNegative = computed(() => neuronDetail.value?.top_input_weights_negative?.slice(0, 4) || []);

const maxAbsWeight = computed(() => {
  const pos = (neuronDetail.value?.top_input_weights_positive || []).map((x: any) => Math.abs(x.weight));
  const neg = (neuronDetail.value?.top_input_weights_negative || []).map((x: any) => Math.abs(x.weight));
  return Math.max(...pos, ...neg, 0.0001);
});

const maxHistCount = computed(() => {
  const counts = neuronDetail.value?.activation_histogram?.counts || [];
  return Math.max(...counts, 1);
});

const selTokIdx = computed(() => {
  if (!selectedNeuron.value || !treeData.value) return null;
  const layer = treeData.value.layers.find((l: any) => l.layer_index === selectedNeuron.value.layer);
  const n = layer?.mlp_neurons_preview?.find((x: any) => x.neuron_index === selectedNeuron.value.neuron_index);
  return n && typeof n.top_token_index === 'number' ? n.top_token_index : null;
});

/* ── API helpers ────────────────────────────────── */
/**
 * Every backend call in this view goes through here.
 *
 * It throws instead of swallowing: a request that never completed must not
 * become an empty result that reads like a measurement. `POST /api/gpt2/*`
 * fails closed -- it answers `status: "unavailable"` with an `error` string
 * when there are no live weights -- so the status and the provenance are
 * both checked here and the backend's own reason is what the user reads.
 */
async function callApi(path: string, params: Record<string, unknown> = {}): Promise<any> {
  const record = await api.pythonCall(path, params) as any;
  if (!record || typeof record !== 'object') {
    throw new Error(`POST /api/${path} returned no payload.`);
  }
  if (!isMeasurement(record)) {
    throw new Error(recordReason(record, path));
  }
  return record;
}

/**
 * A record is a measurement only when the backend both ran it and attested it.
 *
 * `status: "ok"` alone is not enough: the API boundary withholds a record it
 * cannot vouch for by setting `provenance: "unavailable"`, and such a record
 * still carries its numbers and can still say `status: "ok"`. Rendering those
 * numbers would present an unmeasured value as a measurement.
 */
function isMeasurement(record: any): boolean {
  if (!record || typeof record !== 'object') return false;
  if (record.status !== 'ok') return false;
  const provenance = typeof record.provenance === 'string' ? record.provenance.trim().toLowerCase() : '';
  return provenance !== 'unavailable';
}

/** The backend's own words for why nothing was measured. */
function recordReason(record: any, path: string): string {
  const detail = [record?.error, record?.reason]
    .find((value) => typeof value === 'string' && value.trim());
  if (typeof detail === 'string') return detail.trim();
  const status = typeof record?.status === 'string' && record.status ? record.status : 'unknown';
  return `POST /api/${path} did not run (status: ${status}).`;
}

function reasonOf(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  return String(error);
}

/* ── model loading ──────────────────────────────── */
onMounted(async () => {
  loading.value = true;
  modelError.value = null;
  try {
    const arch = await api.gpt2Architecture();
    if (!isMeasurement(arch)) {
      modelError.value = recordReason(arch, 'gpt2/architecture');
      return;
    }
    const layerCount = Number(arch.n_layers);
    if (!Number.isFinite(layerCount) || layerCount <= 0) {
      modelError.value = 'POST /api/gpt2/architecture did not report a layer count.';
      return;
    }
    models.value = [{
      id: 'gpt2-small',
      label: arch.model_name || 'GPT-2',
      family: arch.model_type || 'gpt2',
      layers: layerCount,
      heads: arch.n_heads,
      d_model: arch.d_model,
      d_mlp: arch.d_mlp,
    }];
    selectedModel.value = models.value[0];
  } catch (error) {
    modelError.value = reasonOf(error);
  } finally {
    loading.value = false;
  }
});

/* ── tree loading ───────────────────────────────── */
async function loadTree() {
  if (!selectedModel.value) return;
  loading.value = true;
  treeError.value = null;
  const layers: any[] = [];
  const failures: string[] = [];
  let anyActivations = false;
  for (let li = 0; li < selectedModel.value.layers; li++) {
    try {
      const res = await callApi('gpt2/layer', { layer: li });
      const preview = Array.isArray(res.top_active_neurons) ? res.top_active_neurons : [];
      if (preview.length > 0) anyActivations = true;
      if (Array.isArray(res.str_tokens)) strTokens.value = res.str_tokens;
      layers.push({
        layer_index: res.layer,
        label: res.path || `blocks.${li}`,
        num_attention_heads: res.num_attention_heads,
        num_mlp_neurons: res.num_mlp_neurons,
        residual_stream_dim: res.residual_stream_dim,
        n_params: res.n_params,
        attention_heads_preview: Array.isArray(res.attention_heads) ? res.attention_heads : [],
        mlp_neurons_preview: preview,
        known_circuits: [],
      });
    } catch (error) {
      // One unreadable layer is recorded, not hidden: a partially read model
      // is shown as partial, and a model that could not be read at all is
      // shown as unavailable rather than as an empty tree.
      failures.push(`L${li}: ${reasonOf(error)}`);
    }
  }
  if (!layers.length) {
    treeData.value = null;
    hasActivations.value = false;
    treeError.value = failures.length
      ? `No layer could be read from POST /api/gpt2/layer. ${failures.join(' · ')}`
      : 'POST /api/gpt2/layer returned no layers.';
    loading.value = false;
    return;
  }
  treeData.value = { layers };
  hasActivations.value = anyActivations;
  if (failures.length) {
    treeError.value = `${failures.length} of ${selectedModel.value.layers} layers could not be read. ${failures.join(' · ')}`;
  }
  selectedLayer.value = layers.find((l: any) => l.layer_index === selectedLayer.value?.layer_index) || layers[0];
  loading.value = false;
}

watch(selectedModel, () => {
  treeData.value = null;
  selectedLayer.value = null;
  selectedNeuron.value = null;
  neuronDetail.value = null;
  hasActivations.value = false;
  strTokens.value = [];
  if (selectedModel.value) loadTree();
});

/* ── neuron detail ──────────────────────────────── */
watch(selectedNeuron, () => {
  if (!selectedNeuron.value) return;
  patchResult.value = null;
  patchError.value = null;
  neuronError.value = null;
  neuronDetail.value = null;
  void (async () => {
    try {
      neuronDetail.value = await callApi('gpt2/neuron', {
        layer: selectedNeuron.value.layer,
        neuron_index: selectedNeuron.value.neuron_index,
        component: 'mlp',
        top_k_weights: 16,
      });
    } catch (error) {
      neuronDetail.value = null;
      neuronError.value = reasonOf(error);
    }
  })();
});

/* ── head detail ────────────────────────────────── */
watch(selectedHead, () => {
  if (!selectedHead.value) return;
  headLoading.value = true;
  headError.value = null;
  headDetail.value = null;
  void (async () => {
    try {
      headDetail.value = await callApi('gpt2/attention_head', {
        layer: selectedHead.value.layer,
        head: selectedHead.value.head_index,
      });
    } catch (error) {
      headDetail.value = null;
      headError.value = reasonOf(error);
    } finally {
      headLoading.value = false;
    }
  })();
});

/* ── actions ────────────────────────────────────── */
function selectLayer(layer: any) {
  selectedLayer.value = layer;
  selectedNeuron.value = null;
  neuronDetail.value = null;
  selectedHead.value = null;
  headDetail.value = null;
  headError.value = null;
}

function handleSelectNeuron(sel: { layer: number; neuron_index: number }) {
  selectedNeuron.value = sel;
  selectedHead.value = null;
  headDetail.value = null;
  headError.value = null;
}

function handleSelectHead(layerIndex: number, headIndex: number) {
  selectedHead.value = { layer: layerIndex, head_index: headIndex };
  selectedNeuron.value = null;
  neuronDetail.value = null;
  patchResult.value = null;
}

function onModelChange(e: Event) {
  const val = (e.target as HTMLSelectElement).value;
  selectedModel.value = models.value.find(m => m.id === val) || null;
}

async function handlePatch() {
  if (!selectedNeuron.value) return;
  patchError.value = null;
  patchResult.value = null;
  try {
    patchResult.value = await callApi('gpt2/patch_neuron', {
      layer: selectedNeuron.value.layer,
      neuron_index: selectedNeuron.value.neuron_index,
      patch_value: patchVal.value,
      prompt: 'The Eiffel Tower is in',
    });
  } catch (error) {
    // The patch did not run. Nothing is rendered for it: an unrun patch is
    // not a patch with no effect.
    patchResult.value = null;
    patchError.value = reasonOf(error);
    return;
  }
  try {
    neuronDetail.value = await callApi('gpt2/neuron', {
      layer: selectedNeuron.value.layer,
      neuron_index: selectedNeuron.value.neuron_index,
      component: 'mlp',
      top_k_weights: 16,
    });
  } catch (error) {
    // The patch itself ran; only the refreshed detail could not be read.
    neuronDetail.value = null;
    neuronError.value = reasonOf(error);
  }
}

async function handleRunPrompt() {
  const text = prompt.value.trim();
  if (!text || promptRunning.value) return;
  promptRunning.value = true;
  promptMsg.value = null;
  treeError.value = null;
  try {
    const res = await api.gpt2RunPrompt(text) as any;
    if (!isMeasurement(res)) {
      promptMsg.value = { ok: false, text: `Prompt failed — ${recordReason(res, 'gpt2/run_prompt')}` };
      return;
    }
    const tokenCount = Array.isArray(res.str_tokens) ? res.str_tokens.length : null;
    promptMsg.value = {
      ok: true,
      text: tokenCount === null
        ? 'Prompt ran — the backend reported no token count. Reloading activation map…'
        : `Prompt ran — ${tokenCount} tokens. Reloading activation map…`,
    };
    await loadTree();
    promptMsg.value = treeError.value
      ? { ok: false, text: `Prompt ran, but the activation map could not be read in full. ${treeError.value}` }
      : { ok: true, text: 'Prompt ran — activation map populated.' };
  } catch (e: any) {
    promptMsg.value = { ok: false, text: `Error running prompt: ${e?.message || String(e)}` };
  } finally {
    promptRunning.value = false;
  }
}
</script>
