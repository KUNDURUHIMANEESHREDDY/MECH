# MECH — Mechanistic Interpretability IDE

> **Product definition:** MECH is a local-first research workbench for inspecting, explaining, testing, and experimentally manipulating the internal computations of neural networks—starting with transformer language models.
>
> **Primary users:** mechanistic interpretability researchers, ML engineers, students, and safety researchers.
>
> **Document status:** Product requirements + architecture + implementation contract. Requirements marked **MUST** are release-blocking. **SHOULD** means the default unless there is a documented reason not to implement it. **MAY** indicates an optional capability.

---

## 1. Product vision

MECH should make mechanistic interpretability experiments easier to run, inspect, compare, reproduce, and communicate. It is an IDE for scientific investigation—not a chatbot that claims to explain a model, and not merely a collection of activation visualizations.

A researcher should be able to:

1. Load or register a supported model and see exactly which checkpoint, tokenizer, precision, architecture adapter, and runtime are active.
2. Enter prompts or a dataset, inspect tokenization, run the model, and capture selected internal activations.
3. Explore residual-stream states, attention heads, MLP neurons, individual features, and layer-by-layer logit predictions.
4. Form hypotheses about a behavior and test them with ablation, activation patching, attribution patching, or other explicitly defined interventions.
5. Load or evaluate sparse autoencoders (SAEs), inspect feature activations and examples, and compare feature-based explanations with model behavior.
6. Discover and prune candidate circuits using a defined behavioral metric and methods such as ACDC or attribution-based approximations.
7. Compare experiments, preserve exact settings and evidence, export reproducible reports, and reopen prior work without silently recomputing it.

MECH must distinguish **observation**, **hypothesis**, and **causal evidence**. A heatmap, a high activation, a nearest-neighbor example, or a model-generated feature label is not by itself proof that a component implements a concept.

## 2. Product principles

### 2.1 Scientific integrity

- Every visualization MUST identify its source: model/checkpoint, run ID, layer/module, tensor location, token position, and applicable processing conventions.
- Derived values MUST record the formula, inputs, mask, denominator, normalization, and aggregation used.
- Experiments MUST preserve the baseline, intervention, task metric, data split, seed where applicable, and result.
- MECH MUST show failed, skipped, approximated, or incomplete computations honestly. It must never mark an experiment as validated if the required checks did not run.
- Interpretability claims MUST be phrased proportionally to the evidence. Correlation and descriptive activation are not equivalent to causal necessity or sufficiency.

### 2.2 Local-first and privacy-conscious

- Model execution, activation capture, and experiments SHOULD run locally by default.
- No prompt, model output, activation, dataset, source file, credential, or experiment artifact may be transmitted to an external service without explicit user action and clear disclosure.
- External model-assisted labeling or remote compute MUST be opt-in, scoped, and visible in the run record.
- Hugging Face and provider tokens MUST be read from the user's configured secret store or environment; secrets must not be written into project files, logs, experiment metadata, or crash reports.

### 2.3 Model and backend neutrality

The UI must not assume a single architecture or instrumentation library. MECH SHOULD provide adapters for Hugging Face Transformers, TransformerLens / TransformerBridge, NNsight, and SAELens where their capabilities apply. Unsupported operations must be disabled with a precise reason rather than simulated.

### 2.4 Reproducibility and provenance

- Each experiment is immutable once finalized; users may branch it into a new experiment.
- Large tensors SHOULD be stored in a separate artifact store with metadata in SQLite.
- Reopening an experiment MUST show whether its underlying weights, dataset, adapter, SAE, and software versions are still available.
- Run export MUST include the settings needed to rerun the experiment, excluding secrets.

### 2.5 Useful before GPU access

The application SHOULD launch and provide useful capabilities with CPU-only execution. Model size, memory requirements, unsupported operations, and estimated resource needs must be visible before expensive work begins.

## 3. Goals and non-goals

### Goals

- Provide a coherent GUI for common mechanistic interpretability workflows.
- Make activation inspection and intervention auditable and repeatable.
- Support token-level, layer-level, head-level, neuron-level, and SAE-feature-level exploration when supported by the active adapter.
- Keep the data plane (model execution and tensor processing) separate from the presentation layer.
- Provide a stable internal experiment schema independent of the backend library.
- Support export to JSON/CSV/Parquet/NPZ or safe tensor formats where appropriate, plus Markdown/HTML research reports.
- Make limitations explicit and expose enough raw data for researchers to verify the UI.

### Non-goals

- Automatically discovering a complete, human-readable explanation of an arbitrary model.
- Claiming that an SAE feature maps cleanly to one semantic concept in all contexts.
- Treating neuron, head, or feature labels as objectively ground truth.
- Training frontier-scale language models inside the desktop app.
- Running arbitrary user-supplied Python code with unrestricted privileges by default.
- Sending user prompts or tensors to hosted services without explicit authorization.
- Replacing notebooks, Python, published research, or expert review; MECH complements these workflows.
- Supporting every architecture and every research method in the first release.

## 4. Primary use cases

### UC-01: Inspect a model's next-token computation

1. Select a model checkpoint and runtime device.
2. Enter a prompt and inspect the exact tokenization, including special tokens.
3. Run a forward pass and view next-token logits/probabilities.
4. Open Logit Lens to compare intermediate residual-stream predictions by layer.
5. Inspect the source tensors and processing convention behind any chart.
6. Save the run as an experiment.

### UC-02: Find influential components for a behavior

1. Create a baseline prompt and a corrupted/counterfactual prompt or dataset.
2. Define the target metric (for example, logit difference between target and foil tokens).
3. Capture the needed activations under both conditions.
4. Run activation patching or a supported attribution-based approximation.
5. Sort components by measured metric effect, inspect uncertainty/variance across examples, and run confirmation experiments.
6. Save the full experiment configuration and results.

### UC-03: Explore a neuron or attention head

1. Choose a layer and component.
2. View activation magnitude across token positions and examples.
3. Find top-activating examples from a selected dataset or corpus.
4. Inspect attention patterns or MLP neuron output contribution where supported.
5. Test ablation or patching against a behavioral metric.
6. Record notes and evidence without treating a label as proof.

### UC-04: Explore an SAE feature

1. Select an SAE compatible with the base model, layer, and activation site.
2. Verify metadata and compatibility before loading.
3. Encode captured activations and inspect feature firing rates, activation strength, and example contexts.
4. Inspect decoder directions, approximate logit effects, and reconstruction statistics when supported.
5. Perform ablation or feature intervention in a new experiment.
6. Compare the intervention's effect with the original model and record residual errors.

### UC-05: Discover a candidate circuit

1. Define a task dataset, clean/corrupted pairs, and a scalar metric.
2. Select a supported circuit graph and pruning/attribution method.
3. Run discovery with explicit thresholds, sample count, and compute budget.
4. Display retained nodes/edges, ablation effects, confidence or stability checks, and the metric before/after pruning.
5. Test the discovered circuit on held-out examples and inspect false positives/negatives.
6. Export a circuit artifact and reproducible report.

### UC-06: Reproduce a prior result

Open a saved experiment, inspect provenance and dependency versions, validate local artifacts, and rerun it under the saved configuration. If exact reproduction is impossible, MECH must report the missing component or changed runtime and label the run as non-identical.

---

## 5. System architecture

### 5.1 Recommended topology

MECH SHOULD be a desktop-first application with a TypeScript UI and a separately managed Python research engine. The UI must never import or execute Python model code directly.

```text
┌──────────────────────────────────────────────────────────────┐
│ Desktop shell (Electron)                                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ React + TypeScript UI                                  │  │
│  │ Workbench · Model Library · Activations · SAE · Circuits│  │
│  └────────────────────────────┬───────────────────────────┘  │
│                               │ typed local API / event stream│
│  ┌────────────────────────────▼───────────────────────────┐  │
│  │ Local API service (Python, e.g. FastAPI)                 │  │
│  │ Projects · Experiments · Permissions · Job orchestration │  │
│  └──────────────┬──────────────────────┬────────────────────┘  │
│                 │                      │                       │
│  ┌──────────────▼────────────┐ ┌───────▼───────────────────┐  │
│  │ Research worker process   │ │ Persistence layer         │  │
│  │ PyTorch / model adapters  │ │ SQLite metadata           │  │
│  │ hooks · interventions     │ │ Artifact files / manifests│  │
│  │ SAE · patching · metrics  │ │ Cache / experiment outputs│  │
│  └──────────────┬────────────┘ └───────────────────────────┘  │
│                 │
│  ┌──────────────▼───────────────────────────────────────────┐ │
│  │ Runtime adapters: Transformers · TransformerLens/Bridge   │ │
│  │ NNsight · SAELens · custom adapter protocol               │ │
│  └───────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

The local API and worker may be packaged together for desktop distribution, but worker lifetime, resource control, and crash recovery must remain separate from the UI process. A browser-only mode MAY be provided later, but it must not be required for local GPU access.

### 5.2 Technology selection

The following is a recommended baseline, not permission to force incompatible versions into an existing repository. Verify actual compatibility, pin the tested dependency set, and record migration decisions.

| Layer | Recommended choice | Responsibilities |
|---|---|---|
| Desktop shell | Electron | Window lifecycle, secure application packaging, native filesystem integration through a narrow IPC boundary |
| UI | React + TypeScript + Vite | Research workbench, charts, tables, token viewer, circuit graph |
| UI state | Zustand or Redux Toolkit; choose one | Project selection, current run, views, jobs, transient UI state |
| Typed API | OpenAPI-generated TS client or shared schema generation | Keep frontend/backend contracts aligned |
| Local API | Python + FastAPI | Project, experiment, artifact, job, and capability endpoints |
| Model runtime | PyTorch + Hugging Face Transformers | Model loading and general inference where supported |
| Interpretability adapters | TransformerLens / TransformerBridge, NNsight | Activation capture, hooks, interventions, architecture introspection |
| SAE adapter | SAELens; extensible adapter protocol | SAE metadata, encode/decode, feature analysis and supported interventions |
| Persistence | SQLite + migrations | Metadata, projects, runs, settings, indexes, event records |
| Large artifacts | Local content-addressed files; NumPy/Safetensors/Parquet as appropriate | Activations, metrics, tensors, datasets, plots, exports |
| Graph display | React Flow or equivalent | Circuit graph and dependency/activation graph visualization |
| Charts | Plotly, ECharts, or a project-standard chart library | Activation maps, logit trajectories, feature metrics, patching results |
| Packaging | Electron Builder or Forge + Python bundling strategy | Reproducible installers, runtime discovery, update strategy |

Use adapters rather than importing one interpretability package throughout the application. Since library APIs change, the adapter layer must own compatibility checks and version-specific behavior.

### 5.3 Process boundaries

- **Electron main process:** window lifecycle, restricted filesystem dialogs, and allow-listed IPC only.
- **Renderer process:** UI only. It MUST NOT have unrestricted Node integration or direct shell access.
- **API service:** authenticates local requests, validates schemas, checks project permissions, records operations, manages jobs, and serves metadata/artifacts.
- **Research worker:** loads model weights, runs GPU/CPU operations, handles tensor lifetimes, and emits progress and structured results.
- **Artifact writer:** streams large outputs to temporary files and atomically finalizes them. It must not send multi-gigabyte tensors through JSON or WebSocket messages.

### 5.4 Job execution model

Every expensive computation is a job with a stable `job_id` and a state machine:

`queued → validating → preparing → running → finalizing → succeeded`

Terminal alternatives are `failed`, `cancelled`, `interrupted`, and `blocked`.

Each job must expose:

- job and experiment IDs;
- operation type and readable label;
- model and dataset references;
- creation/start/end timestamps;
- current stage and progress when progress can be measured;
- estimated/observed memory where available;
- cancellation support and whether cancellation is safe at the current stage;
- structured logs with secrets redacted;
- output artifact manifest and any warnings;
- actual status, never a synthetic success state.

Do not make progress percentages up when the underlying operation cannot report meaningful progress. Display a stage label and elapsed time instead.

---

## 6. Core domain model

All persistent entities require schema versions and stable IDs. Timestamps use UTC ISO 8601; durations are numeric milliseconds or seconds with a documented unit.

### 6.1 Project

A project groups models, datasets, experiments, notes, and artifacts.

Minimum fields:

- `id`, `name`, `created_at`, `updated_at`;
- `root_path` (optional; never infer permission to access arbitrary paths from this value);
- project-level settings and default artifact directory;
- labels/tags;
- schema version.

### 6.2 Model reference

- `model_id`;
- provider/registry, repository or local path;
- immutable revision/commit hash where available;
- config and tokenizer identifiers/revisions;
- architecture family and detected module map;
- parameter count and tensor dtype where detectable;
- context window, vocabulary size, layer count, hidden size, attention metadata where available;
- license / usage note and gated status when available;
- adapter selected and adapter version;
- runtime framework versions;
- trust configuration, quantization, device mapping, and weight-processing configuration;
- model file checksums when feasible.

A model name alone is not a reproducibility identifier. A moving branch such as `main` must be resolved to a revision if the source provides one.

### 6.3 Dataset

- `dataset_id`, source, revision or content hash;
- schema, splits, example count, and field mapping;
- preprocessing/tokenization settings;
- privacy classification and local/remote permissions;
- optional `clean`/`corrupted` pair mapping;
- deduplication policy and split manifest;
- dataset license/source citation when known.

MECH must not upload a dataset to create previews or labels unless the user explicitly selects that action.

### 6.4 Experiment

- `experiment_id`, project ID, parent/branch ID;
- title, research question/hypothesis, notes, tags;
- operation type and schema version;
- model, tokenizer, adapter, dataset, and SAE references;
- configuration and metric definitions;
- random seed(s), deterministic flags, runtime/device information;
- created/started/completed timestamps and terminal status;
- baseline references and intervention definitions;
- output artifact manifest;
- structured events, warnings, and errors;
- summary containing observed evidence and limitations.

### 6.5 Artifact

- `artifact_id`, `kind`, URI/path relative to the managed data root;
- content hash, byte size, media/content type;
- schema version, producer job, model/experiment linkage;
- tensor shape, dtype, axis names, axis semantics, and units where relevant;
- retention policy and cache status;
- sensitivity classification.

### 6.6 Observation, hypothesis, result

The data model SHOULD keep these concepts distinct:

- **Observation:** recorded tensor/metric/example, with provenance.
- **Hypothesis:** a human or AI-authored interpretation with author, confidence, rationale, and links to evidence.
- **Result:** outcome of a defined test of a hypothesis, including positive, negative, inconclusive, or invalidated.

AI-generated labels and summaries must be explicitly marked as model-generated hypotheses.

---

## 7. Main application information architecture

### 7.1 Global navigation

Recommended top-level areas:

1. **Workbench** — prompt, model output, tokenization, layer timeline, quick inspection.
2. **Models** — checkpoints, runtime setup, architecture support, resource estimates.
3. **Activations** — capture browser, activation maps, component inspector, saved caches.
4. **Logit Lens** — intermediate logits/probabilities and token-rank trajectories.
5. **Neurons & Heads** — MLP neurons, attention heads, activation examples, ablation.
6. **SAE Lab** — SAE compatibility, feature search, feature examples, reconstruction and intervention.
7. **Circuits** — graph exploration, patching, ACDC pruning, circuit evaluation.
8. **Experiments** — jobs, history, diff/compare, artifacts, reproducibility.
9. **Datasets** — local datasets, prompt pairs, split and metric setup.
10. **Settings** — model cache, device, privacy, API, logs, retention, adapter diagnostics.

The navigation may be simplified in the initial release, but core research capabilities should not be hidden behind a generic chat interface.

### 7.2 Workbench layout

The primary workbench SHOULD use a clear three-region layout:

- **Left rail:** project, model selector, datasets, saved experiments, module navigation.
- **Main canvas:** prompt/output and the currently selected analysis view.
- **Inspector pane:** selected tensor/component/feature details, provenance, numeric values, notes, and available interventions.

The user must be able to resize panels, collapse them, and restore a sensible default layout. Layout state is a user preference, not research data.

### 7.3 Visual design

- Professional, restrained, high-information-density design.
- Light and dark themes; theme selection must not alter data semantics.
- Charts need labelled axes, legends, tooltips, readable numeric values, and accessible color choices.
- Color alone cannot indicate important states; pair it with text, icons, shapes, or patterns.
- Use monospace only for tokens, tensor values, code, IDs, and technical labels where it improves legibility.
- Every view must handle loading, empty, unsupported, stale, failed, and partially available states.
- Avoid decorative animations that make large matrices or graphs harder to inspect.

---

## 8. Model registry and loading

### 8.1 Supported loading sources

MECH SHOULD support:

- model IDs from Hugging Face or another configured registry;
- local model directories and files;
- previously registered checkpoints;
- offline mode using only existing local weights and tokenizers.

Remote model catalog search is a convenience, not a runtime dependency for offline workflows.

### 8.2 Preflight before load

Before loading, show:

- model and revision;
- parameter count, estimated weight memory, expected runtime overhead, and whether activation caching increases memory;
- available CPU/GPU/MPS device(s), free memory when detectable, dtype and quantization options actually supported;
- architecture adapter and available capabilities;
- whether additional files/downloads are needed;
- gated access/licence information if known;
- potential use of `trust_remote_code` and its security consequences.

Resource numbers are estimates and must be labelled as such. Do not promise a model will fit solely because weight size is below reported free memory; activations, temporary buffers, fragmentation, and other processes need headroom.

### 8.3 Model load safety

- `trust_remote_code` MUST default to false. Enabling it requires an explicit warning and confirmation for that model revision.
- Never run downloaded model repository scripts merely to list model information.
- Prefer `safetensors` where available; clearly label legacy pickle-based checkpoints and avoid loading untrusted pickle artifacts unless explicitly permitted.
- Resolve the model revision and record it.
- Authentication tokens must be passed securely to the loader and redacted from error messages.
- A cancelled or failed load must release resources and leave no misleading “loaded” model entry.

### 8.4 Runtime capability handshake

After load, the adapter returns a `ModelCapabilityProfile` describing actual support. Example fields:

```json
{
  "architecture_family": "decoder_transformer",
  "supports_logits": true,
  "supports_hidden_states": true,
  "supports_activation_cache": true,
  "supports_activation_editing": true,
  "supports_attention_patterns": "adapter_dependent",
  "supports_gradient_attribution": false,
  "supports_kv_cache_inspection": false,
  "supports_full_backward": false,
  "supported_interventions": ["ablation", "activation_patch"],
  "unsupported_reasons": {}
}
```

This is illustrative; values must be computed by the adapter and must not be hard-coded across all models.

---

## 9. Instrumentation and model adapter contract

### 9.1 Responsibilities

The adapter layer must normalize how MECH asks for model metadata, performs inference, captures activations, registers interventions, and maps results back to model-native tensors. It must not pretend different model architectures have identical internals.

Minimum conceptual interface:

```python
class InterpretabilityAdapter(Protocol):
    def inspect_model(self) -> ModelMetadata: ...
    def get_capabilities(self) -> ModelCapabilityProfile: ...
    def tokenize(self, inputs: list[str]) -> TokenizationResult: ...
    def forward(self, inputs: ModelInputs, options: ForwardOptions) -> ForwardResult: ...
    def list_activation_sites(self) -> list[ActivationSite]: ...
    def capture_activations(self, request: CaptureRequest) -> ActivationArtifact: ...
    def run_intervention(self, request: InterventionRequest) -> InterventionResult: ...
    def estimate_resources(self, request: ResourceEstimateRequest) -> ResourceEstimate: ...
    def close(self) -> None: ...
```

The real interface should use typed schemas and asynchronous job execution where operations can be long-running. It should not expose raw Python object references over the API.

### 9.2 Adapter implementations

Priority order:

1. Hugging Face Transformers adapter for basic inference, hidden-state extraction, and supported module hooks.
2. TransformerLens / TransformerBridge adapter for hook-aware analysis and patching where the selected version and architecture provide the required functionality.
3. NNsight adapter for supported trace-based activation access/interventions.
4. SAELens adapter for SAE loading, feature encoding/decoding, and compatible integration.
5. Custom adapter protocol for additional research runtimes.

The app should report adapter and package versions in diagnostics. Do not scatter package-specific logic into React components.

### 9.3 Activation site schema

Every activation site should have a stable UI-facing identity and optional backend-native identity:

- `site_id`;
- `display_name` and `canonical_path`;
- `backend_name` / native hook path;
- component category: `residual`, `attention`, `mlp`, `embedding`, `normalization`, `logits`, `other`;
- layer index and optional head/neuron index;
- activation type, known axis names and axis meanings;
- capture/edit/gradient capability flags;
- supported shape constraints and notes.

Axis semantics are critical. A tensor shape such as `[batch, sequence, hidden]` must not be displayed as a generic matrix without labels. GQA/MQA, encoder-decoder, vision, audio, and multimodal architectures require adapter-specific handling.

### 9.4 Hook and intervention lifecycle

- Hooks/interventions must be scoped to one job or an explicitly persisted edited-model session.
- All hooks must be removed in `finally` / equivalent cleanup even if inference throws or cancellation occurs.
- The baseline model must not be mutated unintentionally.
- Interventions must declare whether they alter a transient forward pass, a reusable edited model, or saved weights.
- Saved weight edits must be a distinct advanced action with a separate copy, provenance, and confirmation. They are not part of ordinary activation patching.
- A failed hook must terminate or mark the run invalid; do not quietly run without the requested intervention.

---

## 10. Activation browser

### 10.1 Required capabilities

- Browse activations by layer, site, token position, batch example, and component type.
- View token-aligned heatmaps for tensors where a 2D representation is meaningful.
- Select a cell to inspect exact numeric values and source metadata.
- Compare activations across two inputs or runs using explicit alignment rules.
- Toggle aggregation views such as raw value, absolute value, norm, mean, max, or standardized value; show the formula used.
- Filter and sort component lists by activation magnitude, variance, token position, or selected metric where meaningful.
- Export selected values or tensor slices without exporting every captured tensor.
- Create a saved activation query/preset.

### 10.2 Capture policies

Activation capture can consume substantial RAM, VRAM, and disk space. MECH must support:

- selecting only requested layers/sites;
- token-position slicing;
- batch size and microbatching controls;
- downsampling or aggregating for visualization while retaining raw artifacts when requested;
- per-job capture budgets in bytes or tensor count;
- a cancellation threshold before the system risks exhausting available memory;
- cache eviction based on user-configurable policy, without deleting pinned experiment artifacts.

Never hide whether a visualization uses raw data, a reduced sample, or an aggregate. Raw activation artifacts and display downsampling must be distinguishable.

### 10.3 Comparisons

Comparisons require aligned metadata. MECH should not subtract activations from different checkpoints or token positions without warning. If token sequences differ, show the alignment map and make it possible to inspect unmatched tokens. Any normalization performed before comparison must be explicit and reversible where mathematically possible.

---

## 11. Neuron Inspector

### 11.1 Purpose

Inspect MLP neurons and other architecture-specific units where individual-unit semantics and interventions make sense. Do not force a “neuron” abstraction onto components that do not expose an equivalent unit.

### 11.2 Neuron detail view

Must display, where supported:

- model, layer, block type, neuron index, activation site, tensor dimensions;
- activation distribution over selected examples and token positions;
- top activating contexts from an explicitly selected dataset/corpus;
- activation value distribution, sparsity/frequency, and rank within the relevant layer;
- input/output weight direction or other available parameter views;
- attribution/contribution to selected logits with the method and assumptions labelled;
- baseline vs ablation/patching results;
- researcher notes and links to related experiments.

### 11.3 Top-activating examples

- Rank by an explicitly chosen statistic (e.g. activation at a token, maximum activation per example, mean activation, or positive activation).
- Show the precise triggering token position and full tokenization context.
- Apply consistent preprocessing across examples.
- Avoid presenting overlapping chunks as independent examples without disclosing duplication.
- If examples are from a private corpus, keep them local unless the user separately authorizes an external service.

### 11.4 Ablation and patching

The inspector should offer supported experimental actions:

- zero ablation;
- mean replacement or another documented baseline replacement;
- activation patching from a donor run;
- directional/feature projection ablation only when its definition is explicit.

For each action, show the intervention site, position mask, replacement source, baseline metric, changed metric, and output changes. A change in a single example is exploratory evidence; a robust claim requires evaluation across examples and sensible controls.

---

## 12. Attention Head Explorer

### 12.1 Head-level views

Where the adapter supports attention-pattern capture, display:

- layer and head identifiers;
- query/key token labels;
- attention score/probability matrix and causal/padding masks;
- head output or contribution where available;
- output projection parameters or an equivalent architecture-specific view;
- activation/attention statistics over a dataset;
- examples and token positions responsible for a selected pattern.

### 12.2 Required distinctions

MECH must distinguish attention weights from causal importance. An attention map visualizes one operation and does not, alone, explain the whole behavior. For grouped-query and multi-query attention, the UI must reflect shared key/value heads and avoid showing fictitious independent parameters.

### 12.3 Head interventions

Support head ablation, patching, and supported output replacement. The user must be able to define whether the intervention applies to all tokens, selected token positions, or a token span. Results need a before/after behavioral metric and a record of exactly which tensor was modified.

---

## 13. Logit Lens and residual-stream analysis

### 13.1 Purpose

The Logit Lens view projects intermediate residual-stream representations into vocabulary-logit space to visualize how token predictions evolve through model depth. MECH may later support Tuned Lens or other calibrated projection approaches, but these must be separate methods and never labelled as ordinary Logit Lens.

### 13.2 Required views

- layer-by-layer top-k tokens and logits/probabilities;
- trajectory of a selected target token's logit, rank, and probability;
- comparison between target and foil token logit differences;
- token-position selector and input token viewer;
- layer/time heatmap and sortable table;
- per-layer distribution/entropy when valid for the selected projection;
- optional decomposition or attribution views if the adapter supports them.

### 13.3 Method requirements

- Specify which intermediate representation is projected (e.g. residual stream after a block), which final normalization is applied, and which unembedding matrix is used.
- If layer normalization, centering, weight processing, or other transformations differ from the model's normal forward pass, show the configuration.
- Record whether the displayed distribution is based on exact model logits or an intermediate linear projection.
- Do not compare logits across incompatible tokenizers/vocabularies without a clear warning and mapping.
- Do not call an intermediate top token “the model's final prediction” unless it is the actual final output for that forward pass.

### 13.4 Optional Tuned Lens

Tuned Lens support may be added as a separate adapter-backed capability. A trained translator must be associated with the exact model/revision and layer conventions it was fitted for. The UI must report the lens artifact, validation information, and any mismatch. If no compatible lens exists, show “not available,” not a fabricated approximation.

---

## 14. SAE Lab — sparse autoencoder feature analysis

### 14.1 Objectives

The SAE Lab provides a workflow for loading compatible SAEs, inspecting learned feature activations, retrieving examples, assessing reconstruction, and testing feature interventions. It must support analysis without requiring users to train an SAE inside MECH.

### 14.2 SAE registry and compatibility

Every SAE entry should include:

- SAE ID and immutable source/revision;
- base model and model revision;
- activation site (block/layer/module) and tensor axis convention;
- SAE architecture and dictionary/feature count;
- input dimension and activation preprocessing/normalization;
- training/data metadata and licence where available;
- library and configuration version;
- decoder/encoder parameter artifact identity;
- compatibility status and explanation.

Compatibility MUST be validated before allowing feature analysis. Similar layer names are not enough: base model revision, site, dimensions, preprocessing, and architecture must be checked. User may override only with an explicit warning and recorded decision.

### 14.3 Feature explorer

For a selected feature, show:

- feature index and SAE identity;
- activation distribution, firing frequency, sparsity metric, and activation magnitude statistics;
- top-activating examples with exact token positions;
- token distribution and context examples where available;
- feature decoder direction and supported attribution/logit-effect calculations;
- reconstruction contribution and error metrics when measurable;
- links to nearby features or co-activation patterns, labelled as descriptive similarity;
- any generated description as an unverified hypothesis, never as ground truth.

### 14.4 Feature search

Support filtering by:

- feature index;
- dataset firing frequency;
- top-activation value;
- selected token or phrase occurrence (with tokenization shown);
- user-provided descriptive label, if present;
- similarity/co-activation where computed;
- intervention effect where an experiment exists.

Semantic feature search may use embeddings or generated labels, but must clearly distinguish text search, embedding similarity, and measured activation matching.

### 14.5 SAE metrics

Expose metrics only with definitions and appropriate caveats, including as supported:

- reconstruction loss / explained variance;
- L0-style average active-feature count or another named sparsity measure;
- feature activation frequency;
- dead/rare feature estimates with evaluation dataset and threshold;
- activation norm and reconstruction norm;
- feature ablation effect on a task metric;
- decoder-direction logit attribution where mathematically applicable.

MECH must not imply that low reconstruction error means features are monosemantic or that feature interpretability is established.

### 14.6 Feature interventions

Support only interventions for which the model/SAE adapter can define the intended behavior:

- zero selected feature activation;
- scale or clamp selected feature activations;
- replace activation from a compatible donor run;
- decode edited SAE activations back into the relevant model activation space and inject them when supported.

Record whether the intervention operates in SAE feature space or original activation space. Report reconstruction and intervention artifacts separately. Do not silently alter SAE weights or base model weights.

### 14.7 SAE training (later milestone)

Training and fine-tuning SAEs is an optional later capability. It must have explicit configuration for activation dataset, model/site, training steps, architecture, dictionary size, sparsity objective, optimizer, checkpoint cadence, validation split, and compute budget. Training must run as a cancellable background job with checkpoint/resume support. Do not block the first release on full training support.

---

## 15. Circuit discovery and ACDC pruning engine

### 15.1 Scope

MECH should model a circuit as a task-specific candidate subgraph of computational components and directed information-flow edges, together with the metric used to evaluate whether the subgraph preserves a behavior. It must not imply that the discovered graph is the unique or complete internal algorithm.

### 15.2 Circuit graph schema

A circuit artifact SHOULD contain:

- graph ID and schema version;
- source model/revision and graph-generation method;
- nodes (layer, component type, head/neuron/feature IDs when applicable);
- directed edges and the method used to estimate or define them;
- node/edge scores, thresholds, and ranking method;
- baseline task metric and full-model result;
- pruned-circuit metric and metric delta;
- clean/corrupted dataset references and data split;
- pruning/intervention configuration and thresholds;
- run time, number of forward passes, and compute budget;
- validation/held-out results, warnings, and unsupported operations.

### 15.3 Required discovery workflow

1. Define the task, target behavior, clean/corrupted input construction, and scalar evaluation metric.
2. Check that the task metric actually distinguishes success from failure on the supplied examples.
3. Run baseline and corrupted examples; save outputs and relevant activation caches.
4. Choose an available method: activation patching, attribution patching, ACDC-style edge pruning, or another registered method.
5. Expose all thresholds, node/edge definitions, aggregation choices, and stopping rules.
6. Execute within a bounded budget for examples, forward passes, memory, and time.
7. Display the resulting circuit, metric changes, removed components, and evidence for retained edges.
8. Evaluate the candidate circuit on held-out examples and report failure cases.
9. Save experiment provenance and make the graph exportable.

### 15.4 ACDC-style pruning

The engine must implement a method-specific interface rather than baking one paper's implementation into UI code. The ACDC-style operation should:

- accept a supported model adapter, candidate graph, clean/corrupted dataset, metric, and explicit configuration;
- evaluate candidate edge/component removals according to the selected algorithm;
- record the order of tested removals and metric measurements when feasible;
- support deterministic or seeded runs where the algorithm allows it;
- cache reusable forward-pass results when safe;
- expose the computation budget and support cancellation;
- preserve an unpruned baseline and an immutable trace of changes;
- provide a small, repeatable test case with expected qualitative behavior.

Attribution patching can be offered as a faster approximation, but its results must be labelled approximate. The UI must not merge scores from exact activation patching and first-order attribution into one unlabeled ranking.

### 15.5 Circuit visualizer

- Pan, zoom, select, pin, search, and filter nodes/edges.
- Filter by layer, component class, score threshold, or retained/pruned status.
- Click a node/edge to view numeric metrics, source evidence, and related experiments.
- Show direction and graph legend; use textual labels in addition to color.
- Toggle between full, pruned, and comparison graphs.
- Export graph data to JSON and a static SVG/PNG/PDF view where implemented.
- Support large graphs with progressive rendering or filtering rather than freezing the UI.

### 15.6 Circuit evaluation safeguards

A high scoring circuit on training examples is not enough. MECH SHOULD support held-out evaluation, bootstrapped stability checks or repeated subsampling, threshold sensitivity analysis, and comparison with random/baseline circuits where meaningful. It must show when a circuit was tuned and evaluated on the same examples.

---

## 16. Causal intervention framework

Interventions are a core primitive shared by Neuron Inspector, Head Explorer, SAE Lab, and Circuit Discovery.

### 16.1 Standard intervention types

- **Ablation:** replace a target component's activation with zero or another defined baseline.
- **Activation patching:** replace an activation in a recipient/corrupted run with a corresponding activation from a donor/clean run.
- **Mean replacement:** use a dataset-derived mean with exact source and axis definition.
- **Directional intervention:** add, subtract, project, or remove a defined direction, when supported.
- **Feature-space intervention:** edit an SAE feature activation and decode back to activation space, when supported.
- **Edge/path intervention:** operate only if the adapter exposes the required graph-level mechanism.

### 16.2 Intervention request

Every intervention must explicitly define:

- target model and site;
- donor and recipient run IDs for patching;
- token position/span mask and batch mapping;
- replacement method and tensor source;
- any normalization/scale factors;
- expected shape/dtype/device;
- metric to evaluate;
- comparison baseline;
- deterministic/seed configuration where applicable;
- allowed resource budget.

### 16.3 Result interpretation

Results must include original metric, intervened metric, absolute and relative difference (when meaningful), affected outputs, sample-level results, aggregation, and confidence/stability indicators. If metrics can be negative or near zero, percent-change display must be handled carefully or omitted. Show absolute differences as the reliable baseline.

### 16.4 Controls and validity

The experiment engine SHOULD provide:

- positive and negative controls;
- multiple examples or a configurable sample size;
- independent held-out examples;
- recovery/rescue tests where appropriate;
- token alignment validation;
- sanity checks for metric orientation;
- repeat runs or bootstrap uncertainty estimates where the analysis calls for them.

Ablating a component may damage general model performance or change many unrelated behaviors. The result should not be described as task-specific unless the controls support that conclusion.

---

## 17. Dataset and prompt-pair workbench

### 17.1 Supported inputs

- manually entered prompts;
- newline-delimited JSON / JSONL;
- CSV or Parquet with a user-selected text field;
- supported local dataset connectors;
- clean/corrupted prompt pairs and target/foil token fields.

### 17.2 Dataset preview and validation

Show a preview, row count, schema, missing values, duplicate count, token-length distribution, and potential truncation. Do not silently drop or truncate examples. The user must be able to inspect the exact tokenized input for a chosen row.

### 17.3 Split safety

- Preserve provided train/validation/test split assignments.
- If splitting is requested, record seed, strategy, and proportions.
- Avoid leakage from duplicate or near-duplicate examples where a detector is available.
- A circuit/SAE method must identify which split was used for selecting components and which split was used for evaluation.

### 17.4 Privacy

Dataset paths and sample text can be sensitive. Logs should avoid recording full rows by default. Exporting prompts or samples is a separate user-controlled action.

---

## 18. Metrics and evaluation library

Create a shared, versioned metrics library with input schemas and unit tests. At minimum, consider these metric types when supported by the selected task:

- target token logit;
- target-vs-foil logit difference;
- target probability/rank;
- cross-entropy/loss on labelled text;
- exact match or task accuracy;
- normalized task score with documented denominator;
- activation norm, feature activation, reconstruction loss;
- patching score or recovery fraction with a clearly defined formula;
- latency, forward-pass count, peak observed memory, and artifact size.

Each metric definition must include:

- `metric_id`, name, version, description;
- expected inputs and tensor axes;
- formula and reduction/aggregation;
- direction of improvement (higher, lower, target range, or not applicable);
- masking rules, treatment of invalid samples, and numerical stability strategy;
- units and any normalization;
- test fixtures with known results.

Do not silently switch metric versions inside an old experiment. A change to metric implementation must have a version change or migration note.

---

## 19. Experiment comparison and notebook/report export

### 19.1 Compare runs

The comparison view should align runs by model revision, dataset example, token position, activation site, metric version, and intervention definition. Incompatibilities must be surfaced rather than papered over.

Show:

- configuration diff;
- baseline/intervened metric table;
- per-example distribution rather than only the mean;
- activation/logit/feature comparison where available;
- elapsed time and resource consumption;
- warnings, skipped tests, and failed checks;
- a concise summary of what changed and what cannot be compared.

### 19.2 Report export

Generate a reproducible Markdown/HTML report containing:

- research question and hypothesis;
- model/checkpoint/tokenizer/adapter metadata;
- dataset source and split manifest;
- methods, metric definitions, settings, and seeds;
- plots and tables with captions and units;
- the observed result and uncertainty/stability checks;
- failures, limitations, and unsupported operations;
- links/identifiers for all artifacts and a manifest of file hashes;
- citations/references supplied by the researcher.

Do not write interpretive certainty that exceeds recorded evidence. Reports must differentiate measured result, tentative interpretation, and unanswered question.

### 19.3 Export formats

Depending on artifact type, support JSON, CSV, Parquet, NPZ, Safetensors, Markdown, HTML, SVG, and PNG. Binary formats must be documented with shape, dtype, axes, and schema version. Large exports must show disk-size estimates and require confirmation when they exceed a configurable threshold.

---

## 20. Persistence, caching, and data lifecycle

### 20.1 Storage policy

- SQLite stores metadata, job state, artifact manifests, experiment records, indexes, and small structured outputs.
- Large arrays/tensors are stored as binary artifacts, not in SQLite blobs unless they are intentionally small.
- Artifact files must be written to a temporary destination, validated, then atomically renamed into place.
- Store content hashes for corruption detection and deduplication where useful.
- Use database migrations and backup/restore tests.

### 20.2 Cache versus pinned artifact

A **cache** is disposable and may be evicted; a **pinned artifact** is referenced by a saved experiment and must not be silently deleted. The UI must indicate which is which.

### 20.3 Retention and deletion

Users need controls for cache size, automatic eviction, project deletion, and artifact deletion. Deletion should show affected experiments and dependent artifacts. If an artifact is shared by multiple experiments, delete it only when references and retention policy permit.

### 20.4 Schema evolution

Every persisted dataset has a schema version. Migrations must be transactional and tested from at least the previous supported schema. If migration cannot preserve data, fail safely with a backup/recovery path; never silently discard experiments.

---

## 21. API and event contract

The exact route names may evolve, but the API must remain typed, versioned, and documented. Prefix local API routes with `/api/v1` until a deliberate versioning policy replaces it.

### 21.1 Endpoint families

- `GET /api/v1/health` — service status and engine readiness.
- `GET /api/v1/capabilities` — installed adapters and operation support.
- `GET /api/v1/models` / `POST /api/v1/models/load` — register, inspect, load, or unload models.
- `GET /api/v1/projects` / `POST /api/v1/projects` — project CRUD.
- `GET /api/v1/datasets` / `POST /api/v1/datasets/preview` — dataset registration and preview.
- `POST /api/v1/experiments` — create an experiment.
- `GET /api/v1/experiments/{id}` — experiment metadata and status.
- `POST /api/v1/jobs` — submit a typed research job.
- `GET /api/v1/jobs/{id}` / `POST /api/v1/jobs/{id}/cancel` — job state/cancellation.
- `GET /api/v1/artifacts/{id}` — artifact metadata; stream downloads through a bounded endpoint.
- `POST /api/v1/interventions/validate` — preflight a proposed intervention.
- `GET /api/v1/experiments/{id}/report` — generate or retrieve a report.

These endpoints are a suggested contract, not a mandate to expose unauthenticated network access. The local service should bind to loopback by default and use a per-session capability token for local API calls.

### 21.2 Event stream

Use WebSocket or Server-Sent Events for structured job updates. Event schema:

```json
{
  "event_id": "stable-id",
  "event_version": 1,
  "job_id": "job-id",
  "experiment_id": "experiment-id",
  "timestamp": "2026-01-01T00:00:00Z",
  "type": "job.progress",
  "stage": "capturing_activations",
  "message": "Capturing selected activation sites",
  "progress": null,
  "data": {}
}
```

`progress` may be null when the operation cannot estimate a meaningful fraction. Events must be ordered or carry a sequence number, and clients must recover missed events by requesting current job state.

### 21.3 API rules

- Validate all request/response schemas.
- Use bounded payload sizes and pagination.
- Stream large artifacts instead of serializing them as JSON.
- Apply timeouts, cancellation, and consistent structured errors.
- Redact secrets and avoid echoing prompt contents into generic logs.
- Include request/job correlation IDs in error reports.
- Document error codes, retryability, and whether partial artifacts may exist.

---

## 22. Command-line interface (optional but recommended)

A thin CLI is useful for batch experiments, automation, and integration with notebooks. It should call the same experiment engine and schemas as the UI.

Illustrative commands:

```text
mech doctor
mech models list
mech models inspect <model-id>
mech run forward --model <id> --prompt "..."
mech run logit-lens --model <id> --prompt-file prompt.txt
mech run activation-capture --config capture.yaml
mech run patching --config patching.yaml
mech sae list
mech sae inspect <sae-id>
mech circuits discover --config circuit.yaml
mech experiments list
mech experiments show <experiment-id>
mech experiments export <experiment-id> --format markdown
mech cache status
```

CLI operations must validate config files and write the same structured run records as GUI operations. No command should silently delete caches or modify model weights. Destructive options must be explicit and require confirmation in interactive mode.

---

## 23. Security and safety requirements

### 23.1 Threat model

MECH processes third-party model weights, tokenizers, datasets, possible remote code, experiment files, and potentially sensitive prompts. Risks include arbitrary code execution, path traversal, secret leakage, local API abuse, unsafe deserialization, resource exhaustion, and malicious content embedded in text examples.

### 23.2 Desktop and API security

- Disable Electron renderer Node integration; use context isolation and a narrow preload API.
- Do not expose arbitrary shell commands to the renderer.
- Validate IPC message schemas and allow-list channels.
- Bind local services to `127.0.0.1` or an equivalent loopback interface by default.
- Use a random per-launch session token or equivalent local authorization mechanism for API access.
- Reject unexpected origins and requests without valid authorization.
- Protect against DNS rebinding and cross-origin requests as appropriate for the architecture.
- Do not load arbitrary remote websites in the privileged desktop window.
- Keep dependencies patched and produce a software bill of materials for release artifacts where feasible.

### 23.3 Path and artifact safety

- Normalize and validate paths; resolve symlinks before checking project boundaries.
- Reject path traversal and attempts to write outside an authorized root unless the user explicitly selects a destination through a trusted dialog.
- Use generated artifact filenames rather than interpolating raw user input into paths.
- Do not overwrite an existing artifact without a clear policy and confirmation.
- Validate size, file type, and metadata before parsing uploaded/imported artifacts.

### 23.4 Serialization safety

- Prefer Safetensors and non-executable data formats where possible.
- Avoid arbitrary pickle loading from untrusted sources; if support is essential, gate it behind a clear explicit warning and isolated path.
- Treat datasets and experiment exports as untrusted data, not code or instructions.
- Do not use `eval`, unrestricted dynamic import, shell interpolation, or execution of configuration content to process dataset rows.

### 23.5 Resource-exhaustion safeguards

- Per-job memory, VRAM, disk, time, batch-size, and artifact budgets.
- Cancellation and worker restart for stuck/crashed jobs.
- Queue limits and explicit concurrency controls.
- Warn before expensive full-model activation capture or unbounded feature search.
- Enforce dataset row/token limits configurable by user or project.
- Clean up partial files and GPU resources after failure.

### 23.6 Privacy and telemetry

- Telemetry SHOULD be off by default for sensitive/research builds; if telemetry exists, document every field and provide a clear opt-out.
- Never collect prompts, activations, dataset content, or model weights as generic analytics.
- Crash reports must redact local paths and secrets by default.
- External labeling/search/remote compute requires an explicit data scope and confirmation.

---

## 24. Reliability and recovery

- A failed worker must not crash the UI.
- The UI must distinguish API unavailable, worker unavailable, model not loaded, and operation unsupported.
- Jobs must persist their terminal state and last completed stage.
- On restart, interrupted jobs become `interrupted`; they must not be auto-marked as failed or succeeded without inspection.
- A job may resume only when its method supports checkpoint/resume and the saved state passes validation.
- Artifact writes must be atomic where possible, with temporary artifacts clearly marked and cleaned up after a grace period.
- SQLite operations must use transactions for multi-record changes.
- Provide a diagnostic bundle excluding secrets and private sample data by default.

---

## 25. Performance and scalability

Performance goals should be tested on documented reference hardware, not asserted universally.

### UI

- The interface should remain responsive while model work runs in the worker.
- Virtualize long token lists, feature tables, and example lists.
- Use progressive rendering for graphs and large activation maps.
- Avoid copying large tensor payloads through the frontend; request only visible slices or summaries.

### Research engine

- Support microbatching and selective captures.
- Reuse forward-pass caches only when model, inputs, hooks, precision, and relevant settings match.
- Distinguish cold model-load latency from warm inference latency.
- Profile CPU/GPU transfers and serialization.
- Make compute cost visible before circuit search, wide sweeps, or feature evaluation.

### Initial performance acceptance targets

These are provisional targets to validate on reference hardware:

- A cached metadata view should render within 300 ms after the API response is available.
- UI interaction should remain usable during a long-running model job.
- Cancellation should be acknowledged promptly and reach a safe stopping point within a documented operation-specific bound.
- A small CPU-only reference-model smoke test should run without a GPU.
- Large activation datasets must be paginated or sliced rather than fully loaded into browser memory.

Do not claim a fixed latency for model inference without specifying hardware, model, precision, prompt length, and cache state.

---

## 26. Accessibility and usability

- Keyboard-accessible navigation, dialogs, selectors, tables, and graph controls.
- Visible focus indicators and a logical focus order.
- Screen-reader labels for controls and meaningful charts.
- Text/numeric alternatives for color-coded heatmaps and circuit graphs.
- Respect reduced-motion settings.
- Tooltips must not be the sole place where critical information exists.
- Error messages should say what happened, what data was preserved, and what action the user can take next.
- Scientific terminology should be precise; provide contextual help without obscuring technical details.

---

## 27. Testing strategy

### 27.1 Unit tests

Test:

- schema validation, migrations, and artifact manifests;
- metric calculations and numerical stability;
- activation-axis mapping and token-position alignment;
- hook/intervention lifecycle cleanup;
- adapter capability negotiation;
- SAE compatibility validation;
- circuit graph parsing and serialization;
- cancellation state transitions;
- path validation, secret redaction, and API authorization.

### 27.2 Integration tests

Use small public/reference models or deterministic toy transformers to test:

- model load and inference;
- activation capture at known sites;
- exact shape/axis mapping;
- Logit Lens outputs against a reference calculation;
- activation patching produces the expected directional effect in a controlled fixture;
- SAE encode/decode and feature intervention on a supported toy fixture;
- circuit graph creation, serialization, and pruning on a small graph;
- experiment save, close, reopen, and report export;
- worker crash and restart behavior.

Do not make tests depend on downloading large checkpoints. Download-based tests should be optional, cached, and separately labelled.

### 27.3 Scientific validity tests

- Metric formula checked against hand-computed fixtures.
- Baseline run is preserved before intervention.
- Token alignment failures are detected.
- A deliberately unsupported operation is rejected, not silently approximated.
- Approximate attribution methods are labelled correctly.
- Held-out evaluation cannot accidentally reuse the selection split without a warning.
- Result summaries match underlying numeric artifacts.

### 27.4 UI tests

- loading, empty, unsupported, failed, cancelled, and successful states;
- keyboard navigation and accessible labels;
- selection of token/layer/feature updates the inspector correctly;
- large tables and graphs remain usable;
- stale run selection does not show results from a different experiment;
- UI remains responsive during a long-running worker operation.

### 27.5 End-to-end tests

Run a short workflow from model selection to report export. Include at least one CPU-only path and one supported accelerated path when CI hardware permits.

### 27.6 Release gates

A release candidate MUST pass:

- unit and integration suites;
- static typing/linting for TypeScript and Python;
- dependency/security checks;
- artifact migration and backup/restore tests;
- installer smoke tests on supported operating systems;
- basic offline operation tests;
- no critical open security issues;
- no unlabelled approximation or fabricated result path.

---

## 28. Observability and diagnostics

Use structured logging with `timestamp`, `level`, `component`, `run_id`, `job_id`, `experiment_id`, and an error code where available.

Logs must:

- redact credentials and auth headers;
- avoid full prompt/dataset contents unless the user explicitly enables a research debug log;
- separate user-facing messages from detailed developer diagnostics;
- record runtime/library versions in experiment metadata;
- attach stack traces to local diagnostic logs without automatically transmitting them;
- record warnings when an adapter falls back to a reduced capability mode.

Provide `MECH > Diagnostics` with API/worker status, dependency versions, detected hardware, model cache state, disk usage, adapter capabilities, and recent sanitized errors.

---

## 29. Suggested repository structure

Adapt this to the existing repository instead of blindly replacing it. Preserve useful code, inspect existing conventions, and migrate incrementally.

```text
mech/
├── apps/
│   ├── desktop/
│   │   ├── electron/                 # main process, preload, secure IPC
│   │   └── renderer/                 # React + TypeScript UI
│   └── docs/                         # user and developer documentation
├── packages/
│   ├── ui/                           # shared components and theme
│   ├── api-client/                   # generated/typed API client
│   └── schemas/                      # API contracts, JSON schemas
├── engine/
│   ├── pyproject.toml
│   ├── src/mech/
│   │   ├── api/                      # FastAPI routers and auth
│   │   ├── application/              # use cases and job orchestration
│   │   ├── domain/                   # typed entities and protocols
│   │   ├── adapters/
│   │   │   ├── base.py
│   │   │   ├── transformers.py
│   │   │   ├── transformer_lens.py
│   │   │   ├── nnsight.py
│   │   │   └── saelens.py
│   │   ├── experiments/
│   │   │   ├── forward.py
│   │   │   ├── activation_capture.py
│   │   │   ├── logit_lens.py
│   │   │   ├── interventions.py
│   │   │   ├── metrics.py
│   │   │   ├── patching.py
│   │   │   └── circuits/
│   │   │       ├── graph.py
│   │   │       ├── acdc.py
│   │   │       └── attribution.py
│   │   ├── sae/                      # SAE registry, feature analysis
│   │   ├── storage/                  # SQLite, migrations, artifacts
│   │   ├── security/                 # path policy, secrets, limits
│   │   ├── workers/                  # worker lifecycle and queue
│   │   └── cli/                      # optional CLI
│   └── tests/
├── schemas/
├── scripts/
├── packaging/
├── pyproject.toml or workspace config
├── package.json
├── README.md
├── SECURITY.md
├── CONTRIBUTING.md
└── MECH.md
```

The structure is a target, not an instruction to create empty placeholder files. Each module should exist only when it contains implemented behavior or a meaningful contract with tests.

---

## 30. Implementation plan

Implement vertical slices so each phase produces a working, testable application. Do not build every screen as a mock before connecting a real research engine.

### Phase 0 — Repository and environment audit

- Inspect the existing repository, branch, worktree, scripts, dependency lockfiles, Electron setup, Python environment, and current tests.
- Write down current architecture and preserve existing working modules.
- Detect platform/runtime constraints, especially Python, PyTorch, CUDA, Node, and Electron compatibility.
- Agree on a tested dependency matrix and reproducible install process.
- Define threat model, local storage layout, API contracts, and schema versioning.

**Exit criteria:** repository audit is saved; install/build/test commands are verified or blockers reported; no existing work is overwritten without review.

### Phase 1 — Application skeleton and local engine

- Secure desktop shell and typed renderer-to-main IPC.
- Local API bootstrap, health route, session authorization, and engine readiness diagnostics.
- Python worker lifecycle, cancellation, structured events, and crash handling.
- SQLite migrations, project model, and artifact manifest.
- UI shell with project navigation, job tray, and diagnostics.

**Exit criteria:** application starts from a clean install, reports API/worker status, creates a project, persists it, and shuts down cleanly.

### Phase 2 — Model loading and token workbench

- Model registry and local/Hugging Face loading through a supported adapter.
- Safe preflight and resource estimate.
- Tokenization viewer and forward pass.
- Final logits/top-k output.
- Model unload/resource cleanup.

**Exit criteria:** a small CPU model works offline after caching; model revision and runtime metadata persist; unsupported load settings fail clearly.

### Phase 3 — Activation browser

- Adapter capability handshake and activation site discovery.
- Selective capture with memory budget.
- Activation artifact format and viewer.
- Axis/token alignment and tensor slice export.

**Exit criteria:** activation values match a direct reference computation on deterministic fixtures; large tensors do not freeze the UI.

### Phase 4 — Logit Lens and component inspection

- Intermediate residual extraction and Logit Lens projection.
- Neuron and head views where adapter capabilities support them.
- Source metadata, numeric tables, and research notes.

**Exit criteria:** Logit Lens is validated against a reference formula; unsupported architecture-specific operations are visibly disabled.

### Phase 5 — Interventions and patching

- Intervention schema and preflight validator.
- Ablation and activation patching.
- Baseline/recipient/donor management and comparison view.
- Result persistence and control checks.

**Exit criteria:** tests prove the requested interventions ran, hook cleanup always occurs, and intervention results are reproducible on toy fixtures.

### Phase 6 — SAE Lab

- SAE registry and compatibility checking.
- Load compatible pretrained SAEs via the supported SAELens integration.
- Feature activation, examples, sparsity and reconstruction metrics.
- Feature-level interventions where validated.

**Exit criteria:** at least one model/SAE combination has an end-to-end tested analysis path, with incompatible combinations rejected.

### Phase 7 — Circuit discovery and ACDC engine

- Task metric and clean/corrupted dataset tooling.
- Circuit graph schema and visualizer.
- Baseline activation patching and initial ACDC-style pruning adapter.
- Compute budgets, event trace, held-out validation, and export.

**Exit criteria:** end-to-end circuit discovery on a small reference task; method and limitations documented; no unsupported result is shown as validated.

### Phase 8 — Reproducibility, reports, and hardening

- Run comparison, reports, export manifests, cache retention, and backup/restore.
- Installer builds and diagnostics bundle.
- Security audit, dependency checks, performance tests, accessibility review, and crash recovery.
- Document tested operating systems and hardware constraints.

**Exit criteria:** release gates in Section 27 pass, and the team can reproduce a saved experiment from its manifest on a clean environment subject to availability of the recorded artifacts.

---

## 31. MVP boundary

The initial useful release SHOULD prioritize:

1. secure desktop shell and local API;
2. model loading and exact tokenization;
3. forward-pass results;
4. selective activation capture;
5. Logit Lens;
6. basic neuron/head inspection where supported;
7. simple ablation and activation patching;
8. experiment persistence and comparison;
9. one compatible pretrained SAE path;
10. basic circuit graph and an experimental ACDC-style workflow only after the prerequisites above are reliable.

Do not label the application “production-ready” merely because screens exist. The engine must compute real values, retain provenance, handle failure paths, and test scientific correctness. If ACDC, SAE interventions, or an adapter is not ready, declare that status in the capabilities panel rather than presenting an inert or simulated control.

---

## 32. Definition of done for any feature

A feature is done only when all applicable criteria pass:

- **Functionality:** the complete user workflow performs a real operation.
- **Scientific correctness:** formulas, tensor axes, normalization, intervention site, and assumptions are specified and tested.
- **Provenance:** run inputs, versions, settings, and outputs are recorded.
- **Failure behavior:** unsupported inputs, OOM, cancellation, worker crashes, and invalid artifacts are handled and tested.
- **Security:** files and inputs are validated; no secrets are exposed; remote execution/transfer is explicit.
- **UI quality:** loading, empty, success, partial, error, and unsupported states are implemented.
- **Performance:** large outputs are sliced/paginated; the UI does not block on model work.
- **Accessibility:** keyboard operation and text alternatives exist for important visualizations.
- **Documentation:** user behavior, configuration, limitations, and tested capabilities are documented.
- **Tests:** unit and integration tests exercise the critical path.

---

## 33. Research limitations MECH must communicate

- **Interpretability is incomplete.** Internal features do not always map cleanly to human concepts, and an observed activation pattern may admit multiple explanations.
- **SAEs are approximations.** Reconstruction quality, sparsity, dictionary size, and feature splitting/merging all affect interpretation.
- **Logit Lens is a projection.** Intermediate predictions depend on the projection convention and do not imply that the model has already completed its final computation.
- **Patching can be distribution-shifting.** Replacing activations may produce states the model does not naturally encounter; baseline choice matters.
- **Attribution patching is approximate.** First-order estimates may be inaccurate for nonlinear or large perturbations and should be confirmed with more direct interventions when feasible.
- **Circuits are task- and metric-dependent.** A circuit found for one dataset/metric may not generalize, and graph-pruning methods can miss distributed or redundant pathways.
- **Architecture conventions differ.** Layer names, attention layouts, normalizations, and available hooks vary among models and backends.
- **Memory cost grows quickly.** Capturing all activations across examples and layers may exceed RAM, VRAM, or disk.
- **Labels are hypotheses.** Automatically generated feature or neuron descriptions need empirical validation.

The UI should present relevant caveats at the point where the user views or runs an analysis—not bury every warning in a generic documentation page.

---

## 34. Documentation requirements

Maintain these documents as the project matures:

- `README.md` — overview, screenshots, supported workflows, installation, quick start.
- `docs/architecture.md` — process boundaries, adapter design, persistence, job/event lifecycle.
- `docs/supported-models.md` — tested model families, adapter versions, supported operations and limitations.
- `docs/experiments.md` — how experiments and metrics are represented.
- `docs/logit-lens.md` — exact projection method and known caveats.
- `docs/activation-patching.md` — donor/recipient setup, alignment, metric definitions, limitations.
- `docs/sae-lab.md` — SAE provenance/compatibility and metric definitions.
- `docs/circuit-discovery.md` — ACDC-style configuration, compute cost, validation, and known limitations.
- `docs/security.md` and `SECURITY.md` — trust boundaries and reporting vulnerabilities.
- `docs/data-and-privacy.md` — local storage, remote calls, retention, and deletion.
- `CHANGELOG.md` — user-facing changes and migrations.

Documentation must describe implemented behavior, not aspirational screenshots or planned features as if they already exist.

---

## 35. Official technical references

Use these as upstream technical references and re-check their current APIs before implementation. Pin the actual versions tested by MECH; do not treat this document as a substitute for dependency-specific documentation.

- TransformerLens documentation: <https://transformerlensorg.github.io/TransformerLens/>
- TransformerLens 4.0 migration guide: <https://transformerlensorg.github.io/TransformerLens/content/migrating_to_v4.html>
- TransformerLens activation patching documentation: <https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.patching.html>
- NNsight documentation: <https://nnsight.net/documentation/>
- NNsight quick start: <https://nnsight.net/getting-started/quickstart/>
- SAELens repository and documentation entry point: <https://github.com/decoderesearch/SAELens>
- ACDC implementation / research code: <https://github.com/enkeejunior1/acdc>
- Hugging Face Transformers documentation: <https://huggingface.co/docs/transformers/index>

TransformerLens changed major APIs in 2026: current new integrations should follow the maintained TransformerBridge path rather than assuming the removed legacy `HookedTransformer` API is still available. Isolate all such calls in the adapter and test the exact pinned version. SAELens also has major-version migration considerations; check its current migration guide before implementing training or inference integration.

---

## 36. Final engineering directive

Build MECH as a real scientific instrument. The product is not complete when the interface looks correct; it is complete when researchers can follow every displayed result back to a defined tensor, model revision, dataset, metric, method, and run configuration.

When choosing between a visually impressive feature with ambiguous semantics and a smaller feature with correct mathematics, reproducible artifacts, and transparent limitations, implement the latter first.
