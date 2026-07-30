import { useEffect, useMemo, useState } from "react";
import { getDesktopApi } from "./api";
import type {
  AblationResult,
  AttentionPatternResult,
  CacheShapesResult,
  CircuitTraceResult,
  CorrelatedNeuronsResult,
  DatasetActivationResult,
  ExperimentResult,
  IOIResult,
  LogitLensResult,
  ModelInfo,
  NeuronEvolutionResult,
  NeuronInspectResult,
  NeuronSearchResult,
  PatchingMatrixResult,
  PredictionTraceResult,
  PromptCompareResult,
  PromptResult,
  TokenLookupResult,
} from "../types";

type Tab = "neurons" | "prompts" | "experiments";

function NeuronInspector() {
  const [tab, setTab] = useState<Tab>("prompts");
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [layerIndex, setLayerIndex] = useState(0);
  const [neuronIndex, setNeuronIndex] = useState(0);
  const [tokenIndex, setTokenIndex] = useState(0);
  const [neuronSearchTopK, setNeuronSearchTopK] = useState(20);

  const [promptText, setPromptText] = useState("what is ur name");
  const [promptResult, setPromptResult] = useState<PromptResult | null>(null);
  const [customToken, setCustomToken] = useState("");
  const [customLookup, setCustomLookup] = useState<TokenLookupResult | null>(null);
  const [ioiResult, setIoiResult] = useState<IOIResult | null>(null);
  const [showIoi, setShowIoi] = useState(false);

  const [cacheShapes, setCacheShapes] = useState<CacheShapesResult | null>(null);
  const [ablateLayer, setAblateLayer] = useState(9);
  const [ablateHead, setAblateHead] = useState(9);
  const [ablateResult, setAblateResult] = useState<AblationResult | null>(null);
  const [multiAblateResult, setMultiAblateResult] = useState<AblationResult | null>(null);
  const [attnPatternLayer, setAttnPatternLayer] = useState(0);
  const [attnPatternHead, setAttnPatternHead] = useState(0);
  const [attnPattern, setAttnPattern] = useState<AttentionPatternResult | null>(null);
  const [patchingMatrix, setPatchingMatrix] = useState<PatchingMatrixResult | null>(null);
  const [patchingLoading, setPatchingLoading] = useState(false);
  const [logitLensResult, setLogitLensResult] = useState<LogitLensResult | null>(null);
  const [logitLensLoading, setLogitLensLoading] = useState(false);
  const [selectedLogitLayer, setSelectedLogitLayer] = useState<number | null>(null);
  const [targetToken, setTargetToken] = useState("");
  const [neuronInspectResult, setNeuronInspectResult] = useState<NeuronInspectResult | null>(null);
  const [neuronInspectLoading, setNeuronInspectLoading] = useState(false);
  const [neuronTargetToken, setNeuronTargetToken] = useState("");
  const [neuronSearchResult, setNeuronSearchResult] = useState<NeuronSearchResult | null>(null);
  const [neuronSearchLoading, setNeuronSearchLoading] = useState(false);
  const [correlatedNeuronsResult, setCorrelatedNeuronsResult] = useState<CorrelatedNeuronsResult | null>(null);
  const [correlatedNeuronsLoading, setCorrelatedNeuronsLoading] = useState(false);
  const [neuronEvolutionResult, setNeuronEvolutionResult] = useState<NeuronEvolutionResult | null>(null);
  const [neuronEvolutionLoading, setNeuronEvolutionLoading] = useState(false);
  const [datasetActivationResult, setDatasetActivationResult] = useState<DatasetActivationResult | null>(null);
  const [datasetActivationLoading, setDatasetActivationLoading] = useState(false);
  const [predictionTraceResult, setPredictionTraceResult] = useState<PredictionTraceResult | null>(null);
  const [predictionTraceLoading, setPredictionTraceLoading] = useState(false);
  const [circuitTraceResult, setCircuitTraceResult] = useState<CircuitTraceResult | null>(null);
  const [circuitTraceLoading, setCircuitTraceLoading] = useState(false);
  const [promptB, setPromptB] = useState("The capital of Germany is");
  const [promptCompareResult, setPromptCompareResult] = useState<PromptCompareResult | null>(null);
  const [promptCompareLoading, setPromptCompareLoading] = useState(false);
  const [experimentPrompts, setExperimentPrompts] = useState("When Mary and John went to the store, John gave the bag to\nThe capital of France is\nThe capital of Germany is\nThe Eiffel Tower is located in\nBerlin is the capital of");
  const [experimentResult, setExperimentResult] = useState<ExperimentResult | null>(null);
  const [experimentLoading, setExperimentLoading] = useState(false);
  const [experimentRunPM, setExperimentRunPM] = useState(false);
  const [experimentTopK, setExperimentTopK] = useState(20);

  const api = useMemo(() => {
    try { return getDesktopApi(); }
    catch { return null; }
  }, []);

  useEffect(() => {
    if (!api) { setLoading(false); return; }
    api.modelInfo().then((info) => {
      setModelInfo(info);
      setLoading(false);
    }).catch((e: Error) => {
      setError(e.message);
      setLoading(false);
    });
  }, [api]);

  if (loading) return <div className="inspector-panel"><p>Loading model...</p></div>;
  if (!api) return <div className="inspector-panel"><p>Electron IPC not available.</p></div>;
  if (error) return <div className="inspector-panel"><p>Error: {error}</p></div>;

  const maxLayer = modelInfo ? modelInfo.num_layers - 1 : 11;
  const desktopApi = api;

  async function runPrompt() {
    const result = await desktopApi.prompt.run(promptText, 20);
    setPromptResult(result);
  }

  async function lookupToken() {
    if (!customToken.trim()) return;
    const result = await desktopApi.prompt.tokenLookup(customToken.trim());
    setCustomLookup(result);
  }

  async function runIOI() {
    const result = await desktopApi.prompt.ioi();
    setIoiResult(result);
    setShowIoi(true);
  }

  async function runCacheShapes() {
    const result = await desktopApi.prompt.cacheShapes(promptText);
    setCacheShapes(result);
  }

  async function runAblation() {
    const result = await desktopApi.prompt.ablate(promptText, ablateLayer, ablateHead);
    setAblateResult(result);
  }

  async function runMultiAblation() {
    const result = await desktopApi.prompt.multiAblate(promptText, [[9,9], [9,6], [10,0]]);
    setMultiAblateResult(result);
  }

  async function runAttentionPattern() {
    const result = await desktopApi.prompt.attentionPattern(promptText, attnPatternLayer, attnPatternHead);
    setAttnPattern(result);
  }

  async function runPatchingMatrix() {
    setPatchingLoading(true);
    try {
      const result = await desktopApi.prompt.patchingMatrix(promptText);
      setPatchingMatrix(result);
    } finally {
      setPatchingLoading(false);
    }
  }

  async function runLogitLens() {
    setLogitLensLoading(true);
    setLogitLensResult(null);
    setSelectedLogitLayer(null);
    try {
      const result = await desktopApi.prompt.logitLens(promptText, 5, targetToken.trim() || undefined);
      setLogitLensResult(result);
    } finally {
      setLogitLensLoading(false);
    }
  }

  async function runPredictionTrace() {
    setPredictionTraceLoading(true);
    setPredictionTraceResult(null);
    try {
      const result = await desktopApi.prompt.predictionTrace(promptText, targetToken.trim() || undefined);
      setPredictionTraceResult(result);
    } finally {
      setPredictionTraceLoading(false);
    }
  }

  async function runCircuitTrace() {
    setCircuitTraceLoading(true);
    setCircuitTraceResult(null);
    try {
      const result = await desktopApi.prompt.circuitTrace(promptText, targetToken.trim() || undefined);
      setCircuitTraceResult(result);
    } finally {
      setCircuitTraceLoading(false);
    }
  }

  async function runPromptCompare() {
    setPromptCompareLoading(true);
    setPromptCompareResult(null);
    try {
      const result = await desktopApi.prompt.promptCompare(promptText, promptB, 10);
      setPromptCompareResult(result);
    } finally {
      setPromptCompareLoading(false);
    }
  }

  async function runExperiment() {
    setExperimentLoading(true);
    setExperimentResult(null);
    try {
      const prompts = experimentPrompts.split("\n").map(s => s.trim()).filter(Boolean);
      const config: Record<string, unknown> = { top_k: experimentTopK, run_patching_matrix: experimentRunPM };
      const result = await desktopApi.prompt.runExperiment(prompts, config);
      setExperimentResult(result);
    } finally {
      setExperimentLoading(false);
    }
  }

  async function runDatasetActivation() {
    setDatasetActivationLoading(true);
    setDatasetActivationResult(null);
    try {
      const result = await desktopApi.prompt.datasetActivation(layerIndex, neuronIndex);
      setDatasetActivationResult(result);
    } finally {
      setDatasetActivationLoading(false);
    }
  }

  async function runNeuronEvolution() {
    setNeuronEvolutionLoading(true);
    setNeuronEvolutionResult(null);
    try {
      const result = await desktopApi.prompt.neuronEvolution(promptText, layerIndex, neuronIndex, tokenIndex);
      setNeuronEvolutionResult(result);
    } finally {
      setNeuronEvolutionLoading(false);
    }
  }

  async function runCorrelatedNeurons() {
    setCorrelatedNeuronsLoading(true);
    setCorrelatedNeuronsResult(null);
    try {
      const result = await desktopApi.prompt.correlatedNeurons(promptText, layerIndex, neuronIndex, 20);
      setCorrelatedNeuronsResult(result);
    } finally {
      setCorrelatedNeuronsLoading(false);
    }
  }

  async function runNeuronSearch() {
    setNeuronSearchLoading(true);
    setNeuronSearchResult(null);
    try {
      const result = await desktopApi.prompt.neuronSearch(promptText, layerIndex, neuronSearchTopK);
      setNeuronSearchResult(result);
    } finally {
      setNeuronSearchLoading(false);
    }
  }

  async function runNeuronInspect() {
    setNeuronInspectLoading(true);
    setNeuronInspectResult(null);
    try {
      const result = await desktopApi.prompt.neuronInspect(
        promptText, layerIndex, neuronIndex, neuronTargetToken.trim() || undefined
      );
      setNeuronInspectResult(result);
    } finally {
      setNeuronInspectLoading(false);
    }
  }

  return (
    <div className="inspector-panel">
      <div className="panel-header">
        <h2>Neuron Inspector</h2>
        {modelInfo && (
          <span className="muted">
            {modelInfo.num_layers}L · {modelInfo.num_heads}H · {modelInfo.hidden_dim}d
          </span>
        )}
      </div>

      <nav className="tab-bar">
        {(["neurons", "prompts", "experiments"] as Tab[]).map((t) => (
          <button key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </nav>

      <div className="inspector-controls">
        {tab === "neurons" && (
          <>
            <label>
              <span>Prompt</span>
              <input
                type="text"
                className="prompt-input"
                value={promptText}
                onChange={(e) => setPromptText(e.target.value)}
                placeholder="Enter a prompt..."
              />
            </label>
            <label>Layer <input type="number" min={0} max={maxLayer} value={layerIndex} onChange={(e) => setLayerIndex(Number(e.target.value))} /></label>
            <label>Neuron <input type="number" min={0} max={3071} value={neuronIndex} onChange={(e) => setNeuronIndex(Number(e.target.value))} /></label>
            <label>Token <input type="number" min={0} value={tokenIndex} onChange={(e) => setTokenIndex(Number(e.target.value))} /></label>
            <label>Top-K <input type="number" min={1} max={100} value={neuronSearchTopK} onChange={(e) => setNeuronSearchTopK(Number(e.target.value))} /></label>
            <label>
              <span>Ablate</span>
              <input
                type="text"
                value={neuronTargetToken}
                onChange={(e) => setNeuronTargetToken(e.target.value)}
                placeholder="e.g.  Paris"
                className="target-input"
              />
            </label>
          </>
        )}
        {tab === "experiments" && (
          <>
            <label>
              <span>Prompts (one per line)</span>
              <textarea
                className="prompt-textarea"
                rows={6}
                value={experimentPrompts}
                onChange={(e) => setExperimentPrompts(e.target.value)}
                placeholder="Enter prompts, one per line..."
              />
            </label>
            <label>Top-K <input type="number" min={1} max={100} value={experimentTopK} onChange={(e) => setExperimentTopK(Number(e.target.value))} /></label>
            <label>
              <input type="checkbox" checked={experimentRunPM} onChange={(e) => setExperimentRunPM(e.target.checked)} />
              <span> Patching matrix (IOI prompts)</span>
            </label>
          </>
        )}
        {tab === "prompts" && (
          <>
            <label>
              <span>Prompt A</span>
              <input
                type="text"
                className="prompt-input"
                value={promptText}
                onChange={(e) => setPromptText(e.target.value)}
                placeholder="Enter first prompt..."
              />
            </label>
            <label>
              <span>Prompt B</span>
              <input
                type="text"
                className="prompt-input"
                value={promptB}
                onChange={(e) => setPromptB(e.target.value)}
                placeholder="Enter second prompt..."
              />
            </label>
            <label>
              <span>Token lookup</span>
              <input
                type="text"
                value={customToken}
                onChange={(e) => setCustomToken(e.target.value)}
                placeholder="e.g. Paris"
              />
            </label>
            <label>Layer <input type="number" min={0} max={11} value={ablateLayer} onChange={(e) => setAblateLayer(Number(e.target.value))} /></label>
            <label>Head <input type="number" min={0} max={11} value={ablateHead} onChange={(e) => setAblateHead(Number(e.target.value))} /></label>
            <label>Attn L <input type="number" min={0} max={11} value={attnPatternLayer} onChange={(e) => setAttnPatternLayer(Number(e.target.value))} /></label>
            <label>Attn H <input type="number" min={0} max={11} value={attnPatternHead} onChange={(e) => setAttnPatternHead(Number(e.target.value))} /></label>
            <label>
              <span>Target</span>
              <input
                type="text"
                value={targetToken}
                onChange={(e) => setTargetToken(e.target.value)}
                placeholder="e.g.  Paris"
                className="target-input"
              />
            </label>
          </>
        )}
      </div>

      <div className="inspector-actions">
        {tab === "experiments" && (
          <button className="primary-button" onClick={runExperiment} disabled={experimentLoading}>
            {experimentLoading ? "Running..." : `Run Experiment (${experimentPrompts.split("\n").filter(s => s.trim()).length} prompts)`}
          </button>
        )}
        {tab === "neurons" && (
          <>
            <button onClick={runNeuronSearch} disabled={neuronSearchLoading}>{neuronSearchLoading ? "Searching..." : "Search Neurons"}</button>
            <button onClick={runDatasetActivation} disabled={datasetActivationLoading}>{datasetActivationLoading ? "Scanning..." : "Dataset Profile"}</button>
            <button onClick={runNeuronEvolution} disabled={neuronEvolutionLoading}>{neuronEvolutionLoading ? "Computing..." : "Evolution"}</button>
            <button onClick={runCorrelatedNeurons} disabled={correlatedNeuronsLoading}>{correlatedNeuronsLoading ? "Computing..." : "Correlated"}</button>
            <button className="primary-button" onClick={runNeuronInspect} disabled={neuronInspectLoading}>{neuronInspectLoading ? "Running..." : "Inspect Selected"}</button>
          </>
        )}
        {tab === "prompts" && (
          <>
            <button className="primary-button" onClick={runPrompt}>Run</button>
            <button onClick={lookupToken}>Lookup Token</button>
            <button onClick={runIOI}>IOI Analysis</button>
            <button onClick={runCacheShapes}>Cache Shapes</button>
            <button onClick={runAblation}>Ablate Head</button>
            <button onClick={runMultiAblation}>Multi-Ablate (9.9,9.6,10.0)</button>
            <button onClick={runAttentionPattern}>Attn Pattern</button>
            <button onClick={runPatchingMatrix} disabled={patchingLoading}>{patchingLoading ? "Computing..." : "Patch Matrix"}</button>
            <button onClick={runLogitLens} disabled={logitLensLoading}>{logitLensLoading ? "Computing..." : "Logit Lens"}</button>
            <button onClick={runPredictionTrace} disabled={predictionTraceLoading}>{predictionTraceLoading ? "Tracing..." : "Trace Prediction"}</button>
            <button className="primary-button" onClick={runCircuitTrace} disabled={circuitTraceLoading}>{circuitTraceLoading ? "Tracing..." : "Circuit Trace"}</button>
            <button onClick={runPromptCompare} disabled={promptCompareLoading}>{promptCompareLoading ? "Comparing..." : "Compare Prompts"}</button>
          </>
        )}
      </div>

      <div className="inspector-results">
        {tab === "neurons" && neuronSearchResult && (
          <div className="result-card">
            <h3>Top-{neuronSearchResult.top_k} Neurons — Layer {neuronSearchResult.layer} <span className="muted">(of {neuronSearchResult.d_mlp})</span></h3>
            <p><strong>Prompt:</strong> {neuronSearchResult.prompt}</p>
            <div className="token-row">
              {neuronSearchResult.tokens.map((tok, i) => (
                <span key={i} className="token-badge">{tok}</span>
              ))}
            </div>
            <table className="neuron-search-table">
              <thead>
                <tr><th>Rank</th><th>Neuron</th><th>Max Act</th><th>Mean Act</th><th>Strongest Token</th></tr>
              </thead>
              <tbody>
                {neuronSearchResult.neurons.map((n, i) => (
                  <tr key={n.neuron_index} className={neuronIndex === n.neuron_index ? "selected-row" : ""}>
                    <td className="mono">{i + 1}</td>
                    <td className="mono"><strong>#{n.neuron_index}</strong></td>
                    <td className="mono">{n.max_activation.toFixed(4)}</td>
                    <td className="mono">{n.mean_activation.toFixed(4)}</td>
                    <td>"{n.strongest_token_str}"</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {tab === "neurons" && neuronInspectResult && (
          <div className="result-card neuron-inspect-card">
            <h3>Neuron L{neuronInspectResult.layer} · #{neuronInspectResult.neuron_index} <span className="muted">(of {neuronInspectResult.d_mlp})</span></h3>
            <p><strong>Prompt:</strong> {neuronInspectResult.prompt}</p>
            <div className="token-row">
              {neuronInspectResult.tokens.map((tok, i) => (
                <span key={i} className="token-badge">{tok}</span>
              ))}
            </div>

            <div className="neuron-inspect-stats">
              <div className="stat-box">
                <span className="stat-label">Mean</span>
                <span className="stat-value">{neuronInspectResult.statistics.mean.toFixed(4)}</span>
              </div>
              <div className="stat-box">
                <span className="stat-label">Std</span>
                <span className="stat-value">{neuronInspectResult.statistics.std.toFixed(4)}</span>
              </div>
              <div className="stat-box">
                <span className="stat-label">Max</span>
                <span className="stat-value">{neuronInspectResult.statistics.max.toFixed(4)}</span>
              </div>
              <div className="stat-box">
                <span className="stat-label">Min</span>
                <span className="stat-value">{neuronInspectResult.statistics.min.toFixed(4)}</span>
              </div>
              <div className="stat-box">
                <span className="stat-label">Sparsity</span>
                <span className="stat-value">{(neuronInspectResult.statistics.sparsity * 100).toFixed(1)}%</span>
              </div>
              <div className="stat-box">
                <span className="stat-label">Percentile</span>
                <span className="stat-value">{neuronInspectResult.percentile_within_layer.toFixed(1)}%</span>
              </div>
            </div>

            <p><strong>Strongest token:</strong> "{neuronInspectResult.strongest_token.token_str}" (pos {neuronInspectResult.strongest_token.token_index}, act: {neuronInspectResult.strongest_token.activation.toFixed(4)})</p>

            <div className="neuron-activations">
              <strong>Per-token activations:</strong>
              <div className="neuron-act-bars">
                {neuronInspectResult.token_activations.map((ta) => {
                  const maxAct = neuronInspectResult.statistics.max;
                  const barPct = maxAct > 0 ? (ta.activation / maxAct) * 100 : 0;
                  return (
                    <div key={ta.token_index} className="neuron-act-row" title={`"${ta.token_str}" = ${ta.activation.toFixed(4)}`}>
                      <span className="neuron-act-token">{ta.token_str === "\n" ? "\\n" : ta.token_str || "(space)"}</span>
                      <span className="neuron-act-track">
                        <span className="neuron-act-bar" style={{ width: `${Math.max(2, barPct)}%` }} />
                      </span>
                      <span className="neuron-act-value">{ta.activation.toFixed(3)}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="neuron-histogram">
              <strong>Activation histogram:</strong>
              <div className="neuron-hist-bars">
                {neuronInspectResult.histogram.map((bin, i) => {
                  const maxCount = Math.max(...neuronInspectResult.histogram.map(b => b.count));
                  const barH = maxCount > 0 ? (bin.count / maxCount) * 50 : 0;
                  const mid = ((bin.bin_start + bin.bin_end) / 2).toFixed(2);
                  return (
                    <div key={i} className="neuron-hist-col" title={`[${bin.bin_start.toFixed(2)}, ${bin.bin_end.toFixed(2)}): ${bin.count}`}>
                      <span className="neuron-hist-bar" style={{ height: `${barH}px` }} />
                      <span className="neuron-hist-label">{bin.count}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {neuronInspectResult.ablation_effect && (
              <div className="neuron-ablation">
                <strong>Ablation effect on "{neuronInspectResult.ablation_effect.target_token}":</strong>
                <table className="ablate-table">
                  <thead><tr><th></th><th>Logit</th><th>Δ</th></tr></thead>
                  <tbody>
                    <tr><td><strong>Clean</strong></td><td className="mono">{neuronInspectResult.ablation_effect.clean_logit.toFixed(4)}</td><td></td></tr>
                    <tr><td><strong>Ablated</strong></td><td className="mono">{neuronInspectResult.ablation_effect.ablated_logit.toFixed(4)}</td><td className={`mono ${neuronInspectResult.ablation_effect.delta >= 0 ? "pos" : "neg"}`}>{neuronInspectResult.ablation_effect.delta.toFixed(4)}</td></tr>
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {tab === "neurons" && correlatedNeuronsResult && (
          <div className="result-card">
            <h3>Correlated to L{correlatedNeuronsResult.source.layer} · #{correlatedNeuronsResult.source.neuron_index}</h3>
            <p><strong>Prompt:</strong> {correlatedNeuronsResult.prompt}</p>
            <div className="corr-legend"><span className="muted">r = Pearson correlation &nbsp;|&nbsp; causal = |Δ target| when source is ablated (0–1)</span></div>
            <table className="neuron-search-table">
              <thead>
                <tr><th>r</th><th>Causal</th><th>Layer</th><th>Neuron</th></tr>
              </thead>
              <tbody>
                {correlatedNeuronsResult.correlated_neurons.map((c, i) => {
                  const causalClass = c.causal_score !== undefined
                    ? (c.causal_score > 0.3 ? "pos" : c.causal_score > 0.05 ? "" : "neg")
                    : "";
                  return (
                    <tr key={`${c.layer}-${c.neuron_index}`}>
                      <td className={`mono ${c.correlation >= 0 ? "pos" : "neg"}`}>{c.correlation.toFixed(4)}</td>
                      <td className={`mono ${causalClass}`}>
                        {c.causal_score !== undefined ? c.causal_score.toFixed(4) : "—"}
                      </td>
                      <td className="mono">{c.layer}</td>
                      <td className="mono"><strong>#{c.neuron_index}</strong></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {tab === "neurons" && datasetActivationResult && (
          <div className="result-card">
            <h3>Dataset Profile — L{datasetActivationResult.layer} · #{datasetActivationResult.neuron_index}</h3>
            <p>Tested on <strong>{datasetActivationResult.num_prompts}</strong> prompts &nbsp;|&nbsp;
              Mean activation: <strong>{datasetActivationResult.statistics.mean.toFixed(4)}</strong> &nbsp;|&nbsp;
              Max: <strong>{datasetActivationResult.statistics.max.toFixed(4)}</strong> &nbsp;|&nbsp;
              Std: <strong>{datasetActivationResult.statistics.std.toFixed(4)}</strong></p>
            <div className="dataset-results-scroll">
              <table className="neuron-search-table">
                <thead>
                  <tr><th>#</th><th>Activation</th><th>Trigger</th><th>Prompt</th></tr>
                </thead>
                <tbody>
                  {datasetActivationResult.results.map((r, i) => (
                    <tr key={i}>
                      <td className="mono">{i + 1}</td>
                      <td className="mono">{r.max_activation.toFixed(4)}</td>
                      <td>"{r.trigger_token}"</td>
                      <td className="dataset-prompt">{r.prompt}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {tab === "neurons" && neuronEvolutionResult && (
          <div className="result-card">
            <h3>Evolution — L{neuronEvolutionResult.source.layer} · #{neuronEvolutionResult.source.neuron_index} — token "{neuronEvolutionResult.token_str}"</h3>
            <p><strong>Prompt:</strong> {neuronEvolutionResult.prompt} &nbsp;|&nbsp; Token index: <strong>{neuronEvolutionResult.source.token_index}</strong></p>
            <div className="neuron-evolution-bars">
              {neuronEvolutionResult.layers.map((l) => {
                const maxA = neuronEvolutionResult.max_activation;
                const barH = maxA > 0 ? (l.activation / maxA) * 60 : 0;
                return (
                  <div key={l.layer} className="evolution-bar-group" title={`L${l.layer}: ${l.activation.toFixed(4)}`}>
                    <span className="evolution-bar" style={{ height: `${Math.max(2, barH)}px` }} />
                    <span className="evolution-layer-label">{l.layer}</span>
                    <span className="evolution-value">{l.activation.toFixed(2)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {tab === "prompts" && promptResult && (
          <div className="result-card">
            <h3>Run Prompt</h3>
            <p><strong>Prompt:</strong> {promptResult.prompt}</p>
            <div className="token-row">
              <strong>Tokens:</strong>
              {promptResult.tokens.map((tok, i) => (
                <span key={i} className="token-badge">{tok}</span>
              ))}
            </div>
            {promptResult.top1 && (
              <p><strong>Top-1 token:</strong> "{promptResult.top1.token_str}" (ID: #{promptResult.top1.token_id}, logit: {promptResult.top1.logit.toFixed(3)})</p>
            )}
            <div className="prediction-list">
              <strong>Top-{promptResult.predictions.length} predictions:</strong>
              {promptResult.predictions.map((p, i) => (
                <span key={i} className="prediction-row">
                  "{p.token_str}" logit={p.logit.toFixed(2)} prob={(p.probability * 100).toFixed(1)}%
                </span>
              ))}
            </div>
          </div>
        )}

        {tab === "prompts" && customLookup && (
          <div className="result-card">
            <h3>Token Lookup</h3>
            <p><strong>Token:</strong> {customLookup.token}</p>
            <p><strong>Token ID:</strong> #{customLookup.token_id}</p>
            <p><strong>Detokenized:</strong> "{customLookup.token_str_detokenized}"</p>
            <p><strong>Logit:</strong> <span className={customLookup.logit !== null && customLookup.logit > 0 ? "pos" : "neg"}>{customLookup.logit !== null ? customLookup.logit.toFixed(4) : "N/A"}</span></p>
            {customLookup.note && <p className="muted">{customLookup.note}</p>}
          </div>
        )}

        {tab === "prompts" && showIoi && ioiResult && (
          <div className="result-card">
            <h3>IOI Analysis</h3>
            <div className="ioi-clean">
              <p><strong>Clean:</strong> …{ioiResult.clean_prompt.slice(-40)}</p>
              <p>Mary logit: <span className={ioiResult.clean.mary_logit > 0 ? "pos" : "neg"}>{ioiResult.clean.mary_logit.toFixed(4)}</span></p>
              <p>John logit: <span className={ioiResult.clean.john_logit > 0 ? "pos" : "neg"}>{ioiResult.clean.john_logit.toFixed(4)}</span></p>
              <p>Logit diff (Mary - John): <span className={ioiResult.clean.logit_difference > 0 ? "pos" : "neg"}>{ioiResult.clean.logit_difference.toFixed(4)}</span></p>
              <p>Top-1: "{ioiResult.clean.top1_token_str}" (ID: #{ioiResult.clean.top1_token_id})</p>
              <p>Mary {">"} John: <strong>{ioiResult.clean.mary_greater ? "PASS" : "FAIL"}</strong></p>
            </div>
            <hr />
            <div className="ioi-corr">
              <p><strong>Corrupted:</strong> …{ioiResult.corrupted_prompt.slice(-40)}</p>
              <p>Mary logit: <span className={ioiResult.corrupted.mary_logit > 0 ? "pos" : "neg"}>{ioiResult.corrupted.mary_logit.toFixed(4)}</span></p>
              <p>John logit: <span className={ioiResult.corrupted.john_logit > 0 ? "pos" : "neg"}>{ioiResult.corrupted.john_logit.toFixed(4)}</span></p>
              <p>Logit diff (Mary - John): <span className={ioiResult.corrupted.logit_difference > 0 ? "pos" : "neg"}>{ioiResult.corrupted.logit_difference.toFixed(4)}</span></p>
              <p>Top-1: "{ioiResult.corrupted.top1_token_str}" (ID: #{ioiResult.corrupted.top1_token_id})</p>
              <p>Mary {">"} John: <strong>{ioiResult.corrupted.mary_greater ? "PASS" : "FAIL"}</strong></p>
            </div>
          </div>
        )}

        {tab === "prompts" && cacheShapes && (
          <div className="result-card">
            <h3>Cache Shapes — {cacheShapes.prompt}</h3>
            <p>Tokens: {cacheShapes.tokens.join(" | ")}</p>
            <div className="cache-shapes-scroll">
              {Object.entries(cacheShapes.layers).slice(0, 4).map(([block, hooks]) => (
                <div key={block} className="cache-block">
                  <strong>{block}</strong>
                  <table className="shapes-table">
                    <thead><tr><th>Hook</th><th>Shape</th><th>Min</th><th>Max</th><th>Mean</th></tr></thead>
                    <tbody>
                      {Object.entries(hooks).map(([hname, info]) => (
                        <tr key={hname}>
                          <td className="hook-name">{hname}</td>
                          <td className="mono">[{info.shape.join("×")}]</td>
                          <td className="mono">{info.min.toFixed(4)}</td>
                          <td className="mono">{info.max.toFixed(4)}</td>
                          <td className="mono">{info.mean.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
              {Object.keys(cacheShapes.layers).length > 4 && (
                <p className="muted">… and {Object.keys(cacheShapes.layers).length - 4} more layers</p>
              )}
            </div>
          </div>
        )}

        {tab === "prompts" && ablateResult && (
          <div className="result-card">
            <h3>Ablation: {ablateResult.heads.join(", ")} — z[:,:,head,:] = 0</h3>
            <p><strong>Prompt:</strong> {ablateResult.prompt}</p>
            <table className="ablate-table">
              <thead><tr><th></th><th>Top-1 Token</th><th>ID</th><th>Logit</th><th>Mary logit</th><th>John logit</th></tr></thead>
              <tbody>
                <tr>
                  <td><strong>Clean</strong></td>
                  <td>"{ablateResult.clean.top1_token_str}"</td>
                  <td>#{ablateResult.clean.top1_token_id}</td>
                  <td className="mono">{ablateResult.clean.top1_logit?.toFixed(4) ?? "N/A"}</td>
                  <td className="mono">{ablateResult.clean.mary_logit?.toFixed(4) ?? "N/A"}</td>
                  <td className="mono">{ablateResult.clean.john_logit?.toFixed(4) ?? "N/A"}</td>
                </tr>
                <tr>
                  <td><strong>Ablated</strong></td>
                  <td>"{ablateResult.ablated.top1_token_str}"</td>
                  <td>#{ablateResult.ablated.top1_token_id}</td>
                  <td className="mono">{ablateResult.ablated.top1_logit?.toFixed(4) ?? "N/A"}</td>
                  <td className="mono">{ablateResult.ablated.mary_logit?.toFixed(4) ?? "N/A"}</td>
                  <td className="mono">{ablateResult.ablated.john_logit?.toFixed(4) ?? "N/A"}</td>
                </tr>
              </tbody>
            </table>
            <p>
              Same top-1: <strong className={ablateResult.same_top1_prediction ? "neg" : "pos"}>{ablateResult.same_top1_prediction ? "YES" : "NO"}</strong>
              &nbsp;| Max logit diff: <span className="mono">{ablateResult.max_logit_difference.toFixed(6)}</span>
              &nbsp;| Logits changed: <strong className={ablateResult.logits_changed ? "pos" : "neg"}>{ablateResult.logits_changed ? "YES" : "NO"}</strong>
            </p>
          </div>
        )}

        {tab === "prompts" && multiAblateResult && multiAblateResult !== ablateResult && (
          <div className="result-card">
            <h3>Multi-Ablation: {multiAblateResult.heads.join(", ")}</h3>
            <table className="ablate-table">
              <thead><tr><th></th><th>Top-1</th><th>Mary</th><th>John</th><th>Diff</th></tr></thead>
              <tbody>
                <tr>
                  <td><strong>Clean</strong></td>
                  <td>"{multiAblateResult.clean.top1_token_str}"</td>
                  <td className="mono">{multiAblateResult.clean.mary_logit?.toFixed(4) ?? ""}</td>
                  <td className="mono">{multiAblateResult.clean.john_logit?.toFixed(4) ?? ""}</td>
                  <td className="mono">{multiAblateResult.clean.mary_logit != null && multiAblateResult.clean.john_logit != null ? (multiAblateResult.clean.mary_logit - multiAblateResult.clean.john_logit).toFixed(4) : ""}</td>
                </tr>
                <tr>
                  <td><strong>Ablated</strong></td>
                  <td>"{multiAblateResult.ablated.top1_token_str}"</td>
                  <td className="mono">{multiAblateResult.ablated.mary_logit?.toFixed(4) ?? ""}</td>
                  <td className="mono">{multiAblateResult.ablated.john_logit?.toFixed(4) ?? ""}</td>
                  <td className="mono">{multiAblateResult.ablated.mary_logit != null && multiAblateResult.ablated.john_logit != null ? (multiAblateResult.ablated.mary_logit - multiAblateResult.ablated.john_logit).toFixed(4) : ""}</td>
                </tr>
              </tbody>
            </table>
            <p>
              Same top-1: <strong className={multiAblateResult.same_top1_prediction ? "neg" : "pos"}>{multiAblateResult.same_top1_prediction ? "YES (hook failed)" : "NO (hook works)"}</strong>
              &nbsp;| Max diff: <span className="mono">{multiAblateResult.max_logit_difference.toFixed(6)}</span>
            </p>
          </div>
        )}

        {tab === "prompts" && patchingMatrix && (
          <div className="result-card">
            <h3>Activation Patching Matrix</h3>
            <p><strong>Prompt:</strong> {patchingMatrix.prompt.slice(-50)}</p>
            <p>Clean logit diff (Mary - John): <strong className={patchingMatrix.clean_logit_diff > 0 ? "pos" : "neg"}>{patchingMatrix.clean_logit_diff.toFixed(4)}</strong></p>
            <p>Each cell = |Δ logit diff| when head z is zeroed. <span className="muted">Blue=small, Red=large effect.</span></p>
            <div className="patch-matrix-scroll">
              <table className="patch-matrix">
                <thead>
                  <tr>
                    <th>L\H</th>
                    {Array.from({ length: patchingMatrix.heads }, (_, h) => (
                      <th key={h} className="col-head">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {patchingMatrix.matrix.map((row, l) => (
                    <tr key={l}>
                      <td className="row-head">{l}</td>
                      {row.map((val, h) => {
                        const t = patchingMatrix.vmax > patchingMatrix.vmin
                          ? (val - patchingMatrix.vmin) / (patchingMatrix.vmax - patchingMatrix.vmin)
                          : 0.5;
                        const r = Math.round(30 + t * 200);
                        const g = Math.round(30 + (1 - t) * 200);
                        return (
                          <td
                            key={h}
                            className="patch-cell"
                            style={{ backgroundColor: `rgb(${r}, ${g}, 60)` }}
                            title={`L${l} H${h}: Δ=${val.toFixed(4)}`}
                          >
                            {val.toFixed(2)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {tab === "prompts" && logitLensResult && (
          <div className="logit-lens">
            <h3>Logit Lens</h3>
            <p><strong>Prompt:</strong> {logitLensResult.prompt}</p>
            <div className="token-row">
              {logitLensResult.tokens.map((tok, i) => (
                <span key={i} className="token-badge">{tok}</span>
              ))}
            </div>

            {logitLensResult.category_scores && Object.keys(logitLensResult.category_scores).length > 0 && (
              <div className="category-scores">
                <strong>Semantic categories:</strong>
                <div className="category-bars">
                  {Object.entries(logitLensResult.category_scores)
                    .filter(([, score]) => score > 0.01)
                    .sort(([, a], [, b]) => b - a)
                    .map(([cat, score]) => (
                      <div key={cat} className="category-row">
                        <span className="category-label">{cat}</span>
                        <span className="category-track">
                          <span
                            className="category-bar"
                            style={{ width: `${(score / Math.max(...Object.values(logitLensResult.category_scores))) * 100}%` }}
                          />
                        </span>
                        <span className="category-score">{score.toFixed(2)}</span>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {logitLensResult.target_data && logitLensResult.target_data.length > 0 && (
              <div className="logit-sparkline-section">
                <p><strong>Target:</strong> {targetToken || "(target)"} &mdash; rank progression</p>
                <div className="logit-sparkline">
                  {logitLensResult.target_data.map((td) => {
                    const maxRank = 50257;
                    const barH = Math.max(4, (1 - td.rank / maxRank) * 60);
                    return (
                      <div key={td.layer} className="spark-bar-group" title={`L${td.layer}: rank #${td.rank}, logit=${td.logit.toFixed(2)}`}>
                        <span className="spark-layer-label">{td.layer}</span>
                        <span className="spark-bar" style={{ height: `${barH}px` }} />
                        <span className="spark-rank">#{td.rank}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            <div className="logit-lens-layers">
              {logitLensResult.layers.map((layer) => {
                const maxLogit = Math.max(...layer.predictions.map(p => p.logit));
                const cos = logitLensResult.cosine_similarity_to_final[layer.layer];
                const isSelected = selectedLogitLayer === layer.layer;
                const td = logitLensResult.target_data?.[layer.layer];

                return (
                  <div
                    key={layer.layer}
                    className={`logit-layer-card ${isSelected ? "selected" : ""}`}
                    onClick={() => setSelectedLogitLayer(isSelected ? null : layer.layer)}
                  >
                    <div className="logit-layer-header">
                      <strong>Layer {layer.layer}</strong>
                      <div className="logit-layer-badges">
                        {td && <span className="rank-badge" title={`Rank of target token`}>#{td.rank}</span>}
                        <span className="muted" title="Cosine sim to final residual">cos: {cos.toFixed(4)}</span>
                      </div>
                    </div>
                    <div className="logit-predictions">
                      {layer.predictions.map((p) => {
                        const frac = maxLogit > 0 ? p.logit / maxLogit : 0;
                        const barWidth = Math.max(5, Math.min(100, 50 + frac * 50));
                        const isTarget = td && p.token_id === td.token_id;
                        return (
                          <div key={p.token_id} className={`logit-prediction-row ${isTarget ? "target-row" : ""}`}>
                            <span className="logit-token">"{p.token_str}"</span>
                            <span className="logit-bar-track">
                              <span className="logit-bar" style={{ width: `${barWidth}%` }} />
                            </span>
                            <span className="logit-value">{p.logit.toFixed(2)}</span>
                            <span className="logit-prob">{(p.probability * 100).toFixed(2)}%</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {tab === "prompts" && predictionTraceResult && (
          <div className="result-card">
            <h3>Prediction Trace — "{predictionTraceResult.target_token}"</h3>
            <p><strong>Prompt:</strong> {predictionTraceResult.prompt}</p>
            <p>Final logit: <strong>{predictionTraceResult.final_logit.toFixed(4)}</strong></p>
            <div className="trace-grid">
              <div className="trace-column">
                <strong>Layer</strong>
                {predictionTraceResult.layer_contributions.map((lc) => (
                  <div key={lc.layer} className="trace-row">
                    <span className="mono">{lc.layer}</span>
                    <span className="trace-bar-track" title={`Attn: ${lc.attention_logit.toFixed(2)}`}>
                      <span className="trace-bar attn-bar" style={{ width: `${Math.max(2, (lc.attention_logit + 5) / 10 * 100)}%` }} />
                    </span>
                    <span className="trace-bar-track" title={`MLP: ${lc.mlp_logit.toFixed(2)}`}>
                      <span className="trace-bar mlp-bar" style={{ width: `${Math.max(2, (lc.mlp_logit + 5) / 10 * 100)}%` }} />
                    </span>
                  </div>
                ))}
              </div>
              <div className="trace-legend">
                <span><span className="legend-swatch attn" /> Attn</span>
                <span><span className="legend-swatch mlp" /> MLP</span>
              </div>
            </div>

            {predictionTraceResult.head_contributions && (
              <>
                <p><strong>Top attention heads:</strong></p>
                <table className="neuron-search-table">
                  <thead><tr><th>Layer</th><th>Head</th><th>Logit</th></tr></thead>
                  <tbody>
                    {predictionTraceResult.head_contributions.slice(0, 10).map((hc) => (
                      <tr key={`${hc.layer}-${hc.head}`}>
                        <td className="mono">{hc.layer}</td>
                        <td className="mono"><strong>H{hc.head}</strong></td>
                        <td className={`mono ${hc.logit >= 0 ? "pos" : "neg"}`}>{hc.logit.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}
          </div>
        )}

        {tab === "prompts" && circuitTraceResult && (
          <div className="result-card">
            <h3>Circuit Trace — "{circuitTraceResult.target_token}"</h3>
            <p><strong>Prompt:</strong> {circuitTraceResult.prompt}</p>
            <p>Final logit: <strong>{circuitTraceResult.final_logit.toFixed(4)}</strong></p>
            <p className="muted">
              Examined {circuitTraceResult.circuit_summary.num_heads_examined} heads across {circuitTraceResult.circuit_summary.top_layers_by_attention.length} layers
              &nbsp;|&nbsp; Found {circuitTraceResult.circuit_summary.num_neurons_found} associated neurons
            </p>

            {circuitTraceResult.head_contributions && circuitTraceResult.head_contributions.length > 0 && (
              <>
                <p><strong>Important heads (attn → target):</strong></p>
                <table className="neuron-search-table">
                  <thead><tr><th>Layer</th><th>Head</th><th>Logit</th></tr></thead>
                  <tbody>
                    {circuitTraceResult.head_contributions.slice(0, 8).map((hc) => (
                      <tr key={`${hc.layer}-${hc.head}`}>
                        <td className="mono">{hc.layer}</td>
                        <td className="mono"><strong>H{hc.head}</strong></td>
                        <td className={`mono ${hc.logit >= 0 ? "pos" : "neg"}`}>{hc.logit.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}

            {circuitTraceResult.head_neurons.length > 0 && (
              <>
                <p><strong>Important neurons (head→attended-token→neuron):</strong></p>
                <table className="neuron-search-table">
                  <thead><tr><th>Layer</th><th>Head</th><th>Neuron</th><th>Act.</th><th>Trigger</th><th>H-Logit</th></tr></thead>
                  <tbody>
                    {circuitTraceResult.head_neurons.slice(0, 20).map((hn, i) => (
                      <tr key={i}>
                        <td className="mono">{hn.layer}</td>
                        <td className="mono">H{hn.head}</td>
                        <td className="mono"><strong>#{hn.neuron_index}</strong></td>
                        <td className="mono">{hn.activation.toFixed(3)}</td>
                        <td>"{hn.trigger_token_str}"</td>
                        <td className={`mono ${hn.head_logit >= 0 ? "pos" : "neg"}`}>{hn.head_logit.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}

            {circuitTraceResult.direct_neurons.length > 0 && (
              <>
                <p><strong>Direct neurons (active at target position):</strong></p>
                <table className="neuron-search-table">
                  <thead><tr><th>Layer</th><th>Neuron</th><th>Act.</th></tr></thead>
                  <tbody>
                    {circuitTraceResult.direct_neurons.slice(0, 15).map((dn, i) => (
                      <tr key={i}>
                        <td className="mono">{dn.layer}</td>
                        <td className="mono"><strong>#{dn.neuron_index}</strong></td>
                        <td className="mono">{dn.activation.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}
          </div>
        )}

        {tab === "prompts" && promptCompareResult && (
          <div className="result-card">
            <h3>Prompt Comparison</h3>
            <div className="compare-panels">
              <div className="compare-panel">
                <p><strong>A:</strong> {promptCompareResult.prompt_a}</p>
                <p><strong>Top-1:</strong> "{promptCompareResult.top1_a?.token_str}" (logit: {promptCompareResult.top1_a?.logit.toFixed(3)})</p>
                <div className="prediction-list">
                  {promptCompareResult.predictions_a.slice(0, 5).map((p, i) => (
                    <span key={i} className="prediction-row">"{p.token_str}" {p.logit.toFixed(2)}</span>
                  ))}
                </div>
              </div>
              <div className="compare-panel">
                <p><strong>B:</strong> {promptCompareResult.prompt_b}</p>
                <p><strong>Top-1:</strong> "{promptCompareResult.top1_b?.token_str}" (logit: {promptCompareResult.top1_b?.logit.toFixed(3)})</p>
                <div className="prediction-list">
                  {promptCompareResult.predictions_b.slice(0, 5).map((p, i) => (
                    <span key={i} className="prediction-row">"{p.token_str}" {p.logit.toFixed(2)}</span>
                  ))}
                </div>
              </div>
            </div>

            {promptCompareResult.neuron_diffs.length > 0 && (
              <>
                <p><strong>Top differing neurons (|A-B| at last position):</strong></p>
                <table className="neuron-search-table">
                  <thead><tr><th>Layer</th><th>#</th><th>A</th><th>B</th><th>|Diff|</th></tr></thead>
                  <tbody>
                    {promptCompareResult.neuron_diffs.slice(0, 10).map((nd, i) => (
                      <tr key={i}>
                        <td className="mono">{nd.layer}</td>
                        <td className="mono"><strong>#{nd.neuron_index}</strong></td>
                        <td className="mono">{nd.activation_a.toFixed(3)}</td>
                        <td className="mono">{nd.activation_b.toFixed(3)}</td>
                        <td className="mono">{nd.diff.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}

            {promptCompareResult.attention_diffs.length > 0 && (
              <>
                <p><strong>Top differing attention heads:</strong></p>
                <table className="neuron-search-table">
                  <thead><tr><th>Layer</th><th>Head</th><th>|Pattern A - B|</th></tr></thead>
                  <tbody>
                    {promptCompareResult.attention_diffs.slice(0, 10).map((ad, i) => (
                      <tr key={i}>
                        <td className="mono">{ad.layer}</td>
                        <td className="mono"><strong>H{ad.head}</strong></td>
                        <td className="mono">{ad.diff.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}

            <p><strong>Residual norms:</strong></p>
            <div className="trace-grid">
              <div className="trace-column">
                <strong>Layer</strong>
                {promptCompareResult.residual_norms_a.map((rn, l) => (
                  <div key={l} className="trace-row">
                    <span className="mono">{l}</span>
                    <span className="trace-bar-track" title={`A: ${rn.toFixed(2)}`}>
                      <span className="trace-bar attn-bar" style={{ width: `${Math.min(100, rn * 5)}%` }} />
                    </span>
                    <span className="trace-bar-track" title={`B: ${promptCompareResult.residual_norms_b[l].toFixed(2)}`}>
                      <span className="trace-bar mlp-bar" style={{ width: `${Math.min(100, promptCompareResult.residual_norms_b[l] * 5)}%` }} />
                    </span>
                  </div>
                ))}
              </div>
              <div className="trace-legend">
                <span><span className="legend-swatch attn" /> A</span>
                <span><span className="legend-swatch mlp" /> B</span>
              </div>
            </div>
          </div>
        )}

        {tab === "experiments" && experimentResult && (
          <div className="result-card">
            <h3>Experiment Results</h3>
            <div className="experiment-summary">
              <div className="stat-box"><span className="stat-label">Prompts</span><span className="stat-value">{experimentResult.num_success}/{experimentResult.num_prompts}</span></div>
              <div className="stat-box"><span className="stat-label">Failed</span><span className="stat-value">{experimentResult.num_failed}</span></div>
              <div className="stat-box"><span className="stat-label">Duration</span><span className="stat-value">{experimentResult.duration_seconds.toFixed(1)}s</span></div>
            </div>

            {experimentResult.category_scores && Object.keys(experimentResult.category_scores).length > 0 && (
              <div className="category-scores">
                <strong>Average category scores:</strong>
                <div className="category-bars">
                  {Object.entries(experimentResult.category_scores)
                    .sort(([, a], [, b]) => b - a)
                    .map(([cat, score]) => (
                      <div key={cat} className="category-row">
                        <span className="category-label">{cat}</span>
                        <span className="category-track">
                          <span className="category-bar" style={{ width: `${score * 100}%` }} />
                        </span>
                        <span className="category-score">{score.toFixed(2)}</span>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {experimentResult.patching_matrix && (
              <>
                <p><strong>Average patching matrix ({experimentResult.patching_matrix.count} prompts):</strong></p>
                <div className="patch-matrix-scroll">
                  <table className="patch-matrix">
                    <thead><tr><th>L\H</th>{Array.from({ length: experimentResult.patching_matrix.heads }, (_, h) => <th key={h} className="col-head">{h}</th>)}</tr></thead>
                    <tbody>
                      {experimentResult.patching_matrix.matrix.map((row, l) => (
                        <tr key={l}>
                          <td className="row-head">{l}</td>
                          {row.map((val, h) => {
                            const { vmin, vmax } = experimentResult.patching_matrix!;
                            const t = vmax > vmin ? (val - vmin) / (vmax - vmin) : 0.5;
                            const r = Math.round(30 + t * 200);
                            const g = Math.round(30 + (1 - t) * 200);
                            return <td key={h} className="patch-cell" style={{ backgroundColor: `rgb(${r}, ${g}, 60)` }} title={`L${l} H${h}: ${val.toFixed(4)}`}>{val.toFixed(2)}</td>;
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}

            {experimentResult.head_importance.length > 0 && (
              <>
                <p><strong>Average head importance (top {Math.min(15, experimentResult.head_importance.length)}):</strong></p>
                <table className="neuron-search-table">
                  <thead><tr><th>Layer</th><th>Head</th><th>Mean logit</th><th>Std</th><th>Count</th></tr></thead>
                  <tbody>
                    {experimentResult.head_importance.slice(0, 15).map((h, i) => (
                      <tr key={i}>
                        <td className="mono">{h.layer}</td>
                        <td className="mono"><strong>H{h.head}</strong></td>
                        <td className={`mono ${h.mean_logit >= 0 ? "pos" : "neg"}`}>{h.mean_logit.toFixed(4)}</td>
                        <td className="mono">{h.std_logit.toFixed(4)}</td>
                        <td className="mono">{h.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}

            {experimentResult.neuron_importance.length > 0 && (
              <>
                <p><strong>Average neuron importance (top {Math.min(15, experimentResult.neuron_importance.length)}):</strong></p>
                <table className="neuron-search-table">
                  <thead><tr><th>Layer</th><th>Neuron</th><th>Mean act.</th><th>Std</th><th>Count</th></tr></thead>
                  <tbody>
                    {experimentResult.neuron_importance.slice(0, 15).map((n, i) => (
                      <tr key={i}>
                        <td className="mono">{n.layer}</td>
                        <td className="mono"><strong>#{n.neuron_index}</strong></td>
                        <td className="mono">{n.mean_activation.toFixed(4)}</td>
                        <td className="mono">{n.std_activation.toFixed(4)}</td>
                        <td className="mono">{n.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}

            {experimentResult.top1_summary && (
              <>
                <p><strong>Top-1 distribution ({experimentResult.top1_summary.total} prompts):</strong></p>
                <div className="neuron-act-bars">
                  {Object.entries(experimentResult.top1_summary.distribution).slice(0, 15).map(([tok, cnt]) => {
                    const maxCnt = Math.max(...Object.values(experimentResult.top1_summary.distribution));
                    const pct = maxCnt > 0 ? (cnt / maxCnt) * 100 : 0;
                    return (
                      <div key={tok} className="neuron-act-row">
                        <span className="neuron-act-token">"{tok}"</span>
                        <span className="neuron-act-track">
                          <span className="neuron-act-bar" style={{ width: `${Math.max(2, pct)}%` }} />
                        </span>
                        <span className="neuron-act-value">{cnt}</span>
                      </div>
                    );
                  })}
                </div>
              </>
            )}

            {experimentResult.num_failed > 0 && (
              <details>
                <summary>Errors ({experimentResult.num_failed})</summary>
                <div className="error-list">
                  {experimentResult.errors.map((e, i) => (
                    <p key={i} className="error-item"><strong>{e.prompt}</strong>: {e.error}</p>
                  ))}
                </div>
              </details>
            )}
          </div>
        )}

        {tab === "prompts" && attnPattern && (
          <div className="result-card">
            <h3>Attention Pattern — L{attnPattern.layer} H{attnPattern.head}</h3>
            <p>Source: <code>{attnPattern.source}</code></p>
            <p>Shape: {attnPattern.shape.join("×")} | Min: {attnPattern.min.toFixed(4)} | Max: {attnPattern.max.toFixed(4)} | Mean: {attnPattern.mean.toFixed(4)}</p>
            <p>Tokens: {attnPattern.tokens.join(" | ")}</p>
            <div className="matrix-preview">
              {attnPattern.matrix.map((row, i) => (
                <div key={i} className="matrix-row">
                  <span className="token-label">{attnPattern.tokens[i] ?? i}</span>
                  {row.map((v, j) => (
                    <span key={j} className="cell" style={{ opacity: Math.max(0.05, Math.min(v, 1)) }} title={`q${i}→k${j}: ${v.toFixed(4)}`}>{v.toFixed(2)}</span>
                  ))}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default NeuronInspector;
