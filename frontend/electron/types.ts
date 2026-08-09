export type Theme = "system" | "dark" | "light" | "midnight" | "nord" | "dracula" | "solarized-dark" | "solarized-light";

export type Settings = {
  theme: Theme;
  gpuEnabled: boolean;
  cachePath: string;
  modelPath: string;
  workspacePath: string;
};

export type RecentProject = {
  id: number;
  path: string;
  name: string;
  openedAt: string;
};

export type RecentFile = {
  id: number;
  path: string;
  projectPath: string | null;
  openedAt: string;
};

export type WorkspaceSummary = {
  path: string;
  exists: boolean;
  name: string;
  fileCount: number;
};

export type LogEntry = {
  timestamp: string;
  level: "info" | "warn" | "error";
  message: string;
  context?: Record<string, unknown>;
};

// ---- Neuron Inspector types ----

export type ModelInfo = {
  num_layers: number;
  num_heads: number;
  hidden_dim: number;
  seq_len: number;
  layer_names: string[];
};

export type Statistics = {
  max: number;
  mean: number;
  variance: number;
  sparsity: number;
  min?: number;
  std?: number;
  median?: number;
  l1_norm?: number;
  l2_norm?: number;
  num_elements?: number;
};

export type NeuronInspection = {
  neuron_id: string;
  layer: string;
  layer_index: number;
  neuron_index: number;
  activation: number;
  activation_history?: number[];
  statistics: Statistics;
  top_tokens?: Array<{ token_index: number; token_id: number; activation: number }>;
  description?: string;
};

export type AttentionInspection = {
  head: number;
  num_heads: number;
  matrix: number[][];
  importance: number;
  shape: number[];
  layer: string;
  layer_index: number;
  statistics: Statistics;
  top_connections?: Array<{ query: number; key: number; weight: number }>;
};

export type ResidualInspection = {
  layer: string;
  layer_index: number;
  residual_vector: number[];
  shape: number[];
  statistics: Statistics;
  norm?: number;
  contribution?: number;
};

export type LayerInspection = {
  layer: string;
  layer_index: number;
  layer_type: string;
  attention?: AttentionInspection;
  mlp?: Record<string, unknown>;
  residual?: ResidualInspection;
  neurons?: NeuronInspection[];
  statistics?: Statistics;
  metadata?: Record<string, unknown>;
};

export type TokenInspection = {
  token_index: number;
  token_id: number;
  embedding?: number[];
  embedding_shape?: number[];
  attention?: AttentionInspection;
  residual?: ResidualInspection;
  logits?: number[];
  top_predictions?: Array<{ token_id: number; logit: number; probability: number }>;
  statistics?: Statistics;
};

export type LogitInspection = {
  layer: string;
  layer_index: number;
  logits: number[];
  shape: number[];
  top_tokens: Array<{ token_id: number; logit: number; probability: number }>;
  statistics: Statistics;
};

export type HeadRanking = {
  layer: string;
  layer_index: number;
  metric: string;
  rankings: Array<{ head: number; score: number }>;
};

export type VizDTO = {
  visualization_type: string;
  title: string;
  data: number[][];
  x_labels?: string[];
  y_labels?: string[];
  color_scale?: string;
  metadata?: Record<string, unknown>;
};

export type NeuronApi = {
  modelInfo(): Promise<ModelInfo>;
  inspect(layerIndex: number, neuronIndex: number, tokenIndex?: number): Promise<NeuronInspection>;
  inspectLayer(layerIndex: number, topK?: number, tokenIndex?: number): Promise<NeuronInspection[]>;
  search(layerIndex: number, threshold?: number, topK?: number): Promise<Array<Record<string, unknown>>>;
  stats(layerIndex: number): Promise<Statistics>;
};

export type AttentionApi = {
  inspect(layerIndex: number, headIndex: number, tokenIndex?: number): Promise<AttentionInspection>;
  inspectAll(layerIndex: number, tokenIndex?: number): Promise<AttentionInspection[]>;
  rank(layerIndex: number, metric?: string): Promise<HeadRanking>;
};

export type ResidualApi = {
  inspect(layerIndex: number, tokenIndex?: number): Promise<ResidualInspection>;
  inspectAll(tokenIndex?: number): Promise<ResidualInspection[]>;
};

export type LayerApi = {
  inspect(layerIndex: number, opts?: { neurons?: boolean; attention?: boolean; residual?: boolean; topK?: number; tokenIndex?: number }): Promise<LayerInspection>;
  inspectAll(opts?: { neurons?: boolean; attention?: boolean; residual?: boolean; topK?: number; tokenIndex?: number }): Promise<LayerInspection[]>;
};

export type TokenApi = {
  inspect(tokenIndex: number, opts?: { layerIndex?: number }): Promise<TokenInspection>;
};

export type LogitApi = {
  inspect(layerIndex: number, tokenIndex?: number, topK?: number): Promise<LogitInspection>;
};

export type VizApi = {
  neuron(layerIndex: number, topK?: number, tokenIndices?: number[]): Promise<VizDTO>;
  attention(layerIndex: number, headIndex: number, tokenIndex?: number): Promise<VizDTO>;
  residual(tokenIndex?: number, numDims?: number): Promise<VizDTO>;
  all(layerIndex?: number, headIndex?: number, tokenIndex?: number): Promise<Record<string, VizDTO>>;
};

// ---- Prompt types ----

export type TokenPrediction = {
  token_id: number;
  token_str: string;
  logit: number;
  probability: number;
};

export type TokenLookupResult = {
  token: string;
  token_id: number;
  logit: number | null;
  token_str_detokenized?: string;
  note?: string;
};

export type PromptResult = {
  prompt: string;
  tokens: string[];
  top1: TokenPrediction | null;
  predictions: TokenPrediction[];
};

export type IOIResult = {
  clean_prompt: string;
  corrupted_prompt: string;
  clean: {
    mary_logit: number;
    john_logit: number;
    logit_difference: number;
    mary_greater: boolean;
    top1_token_id: number;
    top1_token_str: string;
  };
  corrupted: {
    mary_logit: number;
    john_logit: number;
    logit_difference: number;
    mary_greater: boolean;
    top1_token_id: number;
    top1_token_str: string;
  };
};

export type CacheShapeEntry = {
  shape: number[];
  dtype: string;
  min: number;
  max: number;
  mean: number;
};

export type CacheShapesResult = {
  prompt: string;
  tokens: string[];
  layers: Record<string, Record<string, CacheShapeEntry>>;
};

export type AblationResult = {
  prompt: string;
  heads: string[];
  clean: {
    top1_token_id: number;
    top1_token_str: string;
    top1_logit: number | null;
    mary_logit?: number | null;
    john_logit?: number | null;
  };
  ablated: {
    top1_token_id: number;
    top1_token_str: string;
    top1_logit: number | null;
    mary_logit?: number | null;
    john_logit?: number | null;
  };
  same_top1_prediction: boolean;
  max_logit_difference: number;
  logits_changed: boolean;
};

export type AttentionPatternResult = {
  prompt: string;
  tokens: string[];
  layer: number;
  head: number;
  shape: number[];
  min: number;
  max: number;
  mean: number;
  matrix: number[][];
  source: string;
};

export type PatchingMatrixResult = {
  prompt: string;
  clean_logit_diff: number;
  matrix: number[][];
  layers: number;
  heads: number;
  vmin: number;
  vmax: number;
};

export type LogitLensPrediction = {
  token_str: string;
  token_id: number;
  logit: number;
  probability: number;
};

export type LogitLensLayer = {
  layer: number;
  predictions: LogitLensPrediction[];
};

export type TargetTokenData = {
  layer: number;
  logit: number;
  probability: number;
  rank: number;
  token_id: number;
};

export type LogitLensMilestones = {
  first_top_100: number | null;
  first_top_10: number | null;
  first_top_5: number | null;
  reached_rank_1: number | null;
};

export type LogitLensJump = {
  from_layer: number | null;
  to_layer: number | null;
  positions: number | null;
};

export type LogitLensInterpretation = {
  summary: string;
  pattern: string;
  milestones: LogitLensMilestones;
  final_rank: number;
  biggest_jump: LogitLensJump | null;
};

export type LogitLensResult = {
  prompt: string;
  tokens: string[];
  layers: LogitLensLayer[];
  cosine_similarity_to_final: number[];
  d_model: number;
  target_data?: TargetTokenData[];
  interpretation?: LogitLensInterpretation;
  category_scores: Record<string, number>;
};

export type TokenActivation = {
  token_index: number;
  token_str: string;
  activation: number;
};

export type HistogramBin = {
  bin_start: number;
  bin_end: number;
  count: number;
};

export type AblationEffect = {
  target_token: string;
  target_id: number;
  clean_logit: number;
  ablated_logit: number;
  delta: number;
  abs_delta: number;
};

export type NeuronSearchEntry = {
  neuron_index: number;
  max_activation: number;
  mean_activation: number;
  strongest_token_index: number;
  strongest_token_str: string;
};

export type CorrelatedNeuronEntry = {
  layer: number;
  neuron_index: number;
  correlation: number;
  abs_correlation: number;
  causal_score?: number;
};

export type CorrelatedNeuronsResult = {
  prompt: string;
  source: { layer: number; neuron_index: number };
  tokens: string[];
  top_k: number;
  correlated_neurons: CorrelatedNeuronEntry[];
};

export type CircuitTraceHeadNeuron = {
  layer: number;
  head: number;
  neuron_index: number;
  activation: number;
  trigger_token_index: number;
  trigger_token_str: string;
  head_logit: number;
};

export type CircuitTraceDirectNeuron = {
  layer: number;
  neuron_index: number;
  activation: number;
  source: string;
};

export type CircuitTraceSummary = {
  num_heads_examined: number;
  num_neurons_found: number;
  top_layers_by_attention: number[];
};

export type CircuitTraceResult = {
  prompt: string;
  target_token: string;
  target_id: number;
  final_logit: number;
  layer_contributions: LayerContribution[];
  head_contributions: HeadContribution[];
  head_neurons: CircuitTraceHeadNeuron[];
  direct_neurons: CircuitTraceDirectNeuron[];
  circuit_summary: CircuitTraceSummary;
};

export type NeuronSearchResult = {
  prompt: string;
  layer: number;
  tokens: string[];
  d_mlp: number;
  top_k: number;
  neurons: NeuronSearchEntry[];
};

export type NeuronEvolutionLayer = {
  layer: number;
  activation: number;
};

export type DatasetActivationEntry = {
  prompt: string;
  max_activation: number;
  trigger_token: string;
  trigger_token_index: number;
};

export type DatasetActivationResult = {
  layer: number;
  neuron_index: number;
  num_prompts: number;
  top_k: number;
  results: DatasetActivationEntry[];
  statistics: {
    mean: number;
    std: number;
    max: number;
  };
};

export type LayerContribution = {
  layer: number;
  attention_logit: number;
  mlp_logit: number;
};

export type HeadContribution = {
  layer: number;
  head: number;
  logit: number;
};

export type PredictionTraceResult = {
  prompt: string;
  target_token: string;
  target_id: number;
  final_logit: number;
  d_model: number;
  layer_contributions: LayerContribution[];
  head_contributions: HeadContribution[] | null;
};

export type NeuronEvolutionResult = {
  prompt: string;
  source: { layer: number; neuron_index: number; token_index: number };
  token_str: string;
  tokens: string[];
  layers: NeuronEvolutionLayer[];
  max_activation: number;
  min_activation: number;
};

export type NeuronInspectResult = {
  prompt: string;
  layer: number;
  neuron_index: number;
  tokens: string[];
  d_mlp: number;
  token_activations: TokenActivation[];
  strongest_token: TokenActivation;
  statistics: {
    mean: number;
    std: number;
    max: number;
    min: number;
    sparsity: number;
  };
  histogram: HistogramBin[];
  percentile_within_layer: number;
  ablation_effect: AblationEffect | null;
};

export type PromptCompareNeuronDiff = {
  layer: number;
  neuron_index: number;
  activation_a: number;
  activation_b: number;
  diff: number;
};

export type PromptCompareAttentionDiff = {
  layer: number;
  head: number;
  diff: number;
};

export type PromptCompareLogitLayer = {
  layer: number;
  predictions: LogitLensPrediction[];
};

export type PromptCompareResult = {
  prompt_a: string;
  prompt_b: string;
  tokens_a: string[];
  tokens_b: string[];
  predictions_a: TokenPrediction[];
  predictions_b: TokenPrediction[];
  top1_a: TokenPrediction | null;
  top1_b: TokenPrediction | null;
  neuron_diffs: PromptCompareNeuronDiff[];
  attention_diffs: PromptCompareAttentionDiff[];
  logit_lens_a: PromptCompareLogitLayer[];
  logit_lens_b: PromptCompareLogitLayer[];
  residual_norms_a: number[];
  residual_norms_b: number[];
  num_layers: number;
  num_heads: number;
};

export type ExperimentHeadEntry = {
  layer: number;
  head: number;
  mean_logit: number;
  std_logit: number;
  count: number;
};

export type ExperimentNeuronEntry = {
  layer: number;
  neuron_index: number;
  mean_activation: number;
  std_activation: number;
  count: number;
};

export type ExperimentPatchingMatrix = {
  matrix: number[][];
  layers: number;
  heads: number;
  count: number;
  vmin: number;
  vmax: number;
};

export type ExperimentTop1Summary = {
  total: number;
  distribution: Record<string, number>;
};

export type ExperimentResult = {
  num_prompts: number;
  num_success: number;
  num_failed: number;
  duration_seconds: number;
  config: Record<string, unknown>;
  errors: { prompt: string; error: string }[];
  patching_matrix?: ExperimentPatchingMatrix;
  head_importance: ExperimentHeadEntry[];
  neuron_importance: ExperimentNeuronEntry[];
  top1_summary: ExperimentTop1Summary;
  category_scores?: Record<string, number>;
};

export type PromptApi = {
  run(prompt: string, topK?: number): Promise<PromptResult>;
  tokenLookup(token: string): Promise<TokenLookupResult>;
  ioi(): Promise<IOIResult>;
  cacheShapes(prompt?: string): Promise<CacheShapesResult>;
  ablate(prompt: string, layer: number, head: number): Promise<AblationResult>;
  multiAblate(prompt?: string, heads?: number[][]): Promise<AblationResult>;
  attentionPattern(prompt: string, layer: number, head: number): Promise<AttentionPatternResult>;
  patchingMatrix(prompt?: string): Promise<PatchingMatrixResult>;
  logitLens(prompt?: string, topK?: number, targetToken?: string): Promise<LogitLensResult>;
  neuronInspect(prompt: string, layer: number, neuron: number, targetToken?: string): Promise<NeuronInspectResult>;
  neuronSearch(prompt: string, layer: number, topK?: number): Promise<NeuronSearchResult>;
  correlatedNeurons(prompt: string, layer: number, neuron: number, topK?: number): Promise<CorrelatedNeuronsResult>;
  neuronEvolution(prompt: string, layer: number, neuron: number, tokenIndex: number): Promise<NeuronEvolutionResult>;
  datasetActivation(layer: number, neuron: number, prompts?: string[], topK?: number): Promise<DatasetActivationResult>;
  predictionTrace(prompt: string, targetToken?: string, layerForHeads?: number): Promise<PredictionTraceResult>;
  circuitTrace(prompt: string, targetToken?: string): Promise<CircuitTraceResult>;
  promptCompare(promptA: string, promptB: string, topK?: number): Promise<PromptCompareResult>;
  runExperiment(prompts: string[], config?: Record<string, unknown>): Promise<ExperimentResult>;
};

export type DesktopApi = {
  ping(): Promise<{ ok: boolean; storage: string }>;
  getApiKey(): Promise<string>;
  httpRequest(request: {
    method?: string;
    path?: string;
    headers?: Record<string, string>;
    body?: string;
  }): Promise<{ status: number; statusText: string; headers: Record<string, string>; body: string }>;
  getSettings(): Promise<Settings>;
  updateSettings(settings: Settings): Promise<Settings>;
  listRecentProjects(limit?: number): Promise<RecentProject[]>;
  addRecentProject(projectPath: string, name?: string): Promise<RecentProject>;
  chooseProject(): Promise<RecentProject | null>;
  listRecentFiles(limit?: number): Promise<RecentFile[]>;
  addRecentFile(filePath: string, projectPath?: string): Promise<RecentFile>;
  describeWorkspace(workspacePath: string): Promise<WorkspaceSummary>;
  chooseCachePath(): Promise<string | null>;
  listLogs(): Promise<LogEntry[]>;
  onLogEntry(callback: (entry: LogEntry) => void): () => void;
  // Neuron Inspector API — all real model inference via GPT2Model
  modelInfo(): Promise<ModelInfo>;
  prompt: PromptApi;
};
