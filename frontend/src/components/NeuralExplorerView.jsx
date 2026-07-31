import React, { useState, useCallback, useEffect } from 'react';

const MODELS = [
  { id: 'gpt2-small',  label: 'GPT-2 Small',   family: 'gpt2',    layers: 12, heads: 12 },
  { id: 'gpt2-medium', label: 'GPT-2 Medium',  family: 'gpt2',    layers: 24, heads: 16 },
  { id: 'gemma-2b',    label: 'Gemma 2B',       family: 'gemma',   layers: 18, heads: 8  },
  { id: 'tinyllama',   label: 'TinyLlama 1.1B', family: 'llama',   layers: 22, heads: 32 },
  { id: 'mistral-7b',  label: 'Mistral 7B',     family: 'mistral', layers: 32, heads: 32 },
];

const TIER_COLORS = { Gold: '#f5c518', Silver: '#aaa', Bronze: '#cd7f32', 'Needs Investigation': '#e55' };

function TierBadge({ tier }) {
  return (
    <span style={{
      background: TIER_COLORS[tier] || '#555', color: tier === 'Gold' ? '#111' : '#fff',
      borderRadius: 4, padding: '1px 8px', fontSize: 11, fontWeight: 700, marginLeft: 6}}>{tier}</span>
  );
}

function ScoreBar({ value, max = 1.0, color = '#7c6af7' }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div style={{ background: 'var(--bg)', borderRadius: 4, height: 6, width: '100%', margin: '3px 0' }}>
      <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 4, transition: 'width 0.4s' }} />
    </div>
  );
}

function Chip({ label, color = '#7c6af7' }) {
  return (
    <span style={{
      background: color + '22', color, border: `1px solid ${color}55`,
      borderRadius: 12, padding: '2px 10px', fontSize: 11, marginRight: 4, marginBottom: 4, display: 'inline-block'}}>{label}</span>
  );
}

// ── Column 1: Model Tree ───────────────────────────────────────────────────────

function ModelTree({ selectedModel, onSelectModel, selectedLayer, onSelectLayer, selectedNeuron, onSelectNeuron, treeData }) {
  const [expandedLayers, setExpandedLayers] = useState(new Set());

  const toggleLayer = (idx) => {
    setExpandedLayers(prev => {
      const next = new Set(prev);
      next.has(idx) ? next.delete(idx) : next.add(idx);
      return next;
    });
  };

  return (
    <div style={{ width: 220, minWidth: 180, background: '#0d0d1a', borderRight: '1px solid #2a2a4a', overflowY: 'auto', padding: '12px 0', display: 'flex', flexDirection: 'column' }}>
      {/* Model selector */}
      <div style={{ padding: '0 12px 12px', borderBottom: '1px solid #2a2a4a' }}>
        <div style={{ fontSize: 10, color: 'var(--text-dim)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 }}>Model</div>
        <select
          id="neural-explorer-model-select"
          value={selectedModel?.id || ''}
          onChange={e => onSelectModel(MODELS.find(m => m.id === e.target.value))}
          style={{ width: '100%', background: 'var(--bg)', border: '1px solid #3a3a6a', color: inherit, borderRadius: 6, padding: '6px 8px', fontSize: 12 }}
        >
          {MODELS.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
        </select>
      </div>

      {/* Layer list */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {treeData?.layers?.map(layer => (
          <div key={layer.layer_index}>
            <button
              id={`layer-btn-${layer.layer_index}`}
              onClick={() => { onSelectLayer(layer); toggleLayer(layer.layer_index); }}
              style={{
                width: '100%', textAlign: 'left', padding: '7px 14px',
                background: selectedLayer?.layer_index === layer.layer_index ? '#1e1e38' : 'transparent',
                border: 'none', color: '#c8c8ff', fontSize: 12, cursor: 'pointer',
                borderLeft: selectedLayer?.layer_index === layer.layer_index ? '3px solid #7c6af7' : '3px solid transparent',
                display: 'flex', alignItems: 'center', gap: 6}}
            >
              <span style={{ display: 'inline-flex', color: 'var(--text-dim)' }}>
                <svg width="8" height="8" viewBox="0 0 8 8" style={{ transform: expandedLayers.has(layer.layer_index) ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s' }}>
                  <polygon points="2,0 8,4 2,8" fill="currentColor" />
                </svg>
              </span>
              <span>Layer {layer.layer_index}</span>
              {layer.known_circuits.length > 0 && (
                <span style={{ marginLeft: 'auto', width: 6, height: 6, borderRadius: '50%', background: '#7c6af7' }} title="Part of known circuit" />
              )}
            </button>

            {expandedLayers.has(layer.layer_index) && (
              <div style={{ paddingLeft: 24 }}>
                {layer.mlp_neurons_preview?.slice(0, 16).map(n => (
                  <button
                    key={n.neuron_index}
                    id={`neuron-btn-${layer.layer_index}-${n.neuron_index}`}
                    onClick={() => onSelectNeuron({ layer: layer.layer_index, neuron_index: n.neuron_index })}
                    style={{
                      display: 'block', width: '100%', textAlign: 'left',
                      padding: '3px 8px', background: 'transparent', border: 'none',
                      color: selectedNeuron?.layer === layer.layer_index && selectedNeuron?.neuron_index === n.neuron_index ? '#b0a0ff' : '#6060a0',
                      fontSize: 11, cursor: 'pointer'}}
                  >
                    N{n.neuron_index}
                  </button>
                ))}
                {layer.mlp_neurons_preview?.length > 0 && (
                  <span style={{ fontSize: 10, color: '#555', paddingLeft: 8 }}>
                    +{layer.num_mlp_neurons - 16} more…
                  </span>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Column 2: Layer Detail ─────────────────────────────────────────────────────

function LayerDetail({ layer, selectedNeuron, onSelectNeuron }) {
  if (!layer) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#444', fontSize: 14 }}>
        Select a layer to explore
      </div>
    );
  }
  return (
    <div style={{ width: 260, background: '#0f0f20', borderRight: '1px solid #2a2a4a', overflowY: 'auto', padding: 16 }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: '#b0a0ff', marginBottom: 12 }}>
        Layer {layer.layer_index}
        <span style={{ fontSize: 10, color: 'var(--text-dim)', fontWeight: 400, marginLeft: 8 }}>d={layer.residual_stream_dim}</span>
      </div>

      {layer.known_circuits.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          {layer.known_circuits.map(c => <Chip key={c} label={c.replace('_', ' ')} color="#7c6af7" />)}
        </div>
      )}

      {/* Attention heads */}
      <div style={{ fontSize: 10, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>
        Attention Heads ({layer.num_attention_heads})
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 4, marginBottom: 16 }}>
        {layer.attention_heads_preview?.map(h => (
          <div
            key={h.head_index}
            title={h.known_role || `Head ${h.head_index}`}
            style={{
              padding: '4px 2px', borderRadius: 4, textAlign: 'center', fontSize: 10,
              background: h.is_induction_head ? '#2a1a5a' : '#1a1a2e',
              border: h.is_induction_head ? '1px solid #7c6af7' : '1px solid #2a2a4a',
              color: h.is_induction_head ? '#b0a0ff' : '#6060a0',
              cursor: 'default'}}
          >
            H{h.head_index}
          </div>
        ))}
      </div>

      {/* MLP neurons */}
      <div style={{ fontSize: 10, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>
        MLP Neurons ({layer.num_mlp_neurons})
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 3 }}>
        {layer.mlp_neurons_preview?.slice(0, 32).map(n => (
          <button
            key={n.neuron_index}
            id={`layer-detail-neuron-${layer.layer_index}-${n.neuron_index}`}
            onClick={() => onSelectNeuron({ layer: layer.layer_index, neuron_index: n.neuron_index })}
            style={{
              padding: '4px 2px', borderRadius: 4, textAlign: 'center', fontSize: 10,
              background: selectedNeuron?.layer === layer.layer_index && selectedNeuron?.neuron_index === n.neuron_index
                ? '#3a2a6a' : '#1a1a2e',
              border: selectedNeuron?.layer === layer.layer_index && selectedNeuron?.neuron_index === n.neuron_index
                ? '1px solid #7c6af7' : '1px solid #252540',
              color: '#8080c0', cursor: 'pointer'}}
          >
            {n.neuron_index}
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Column 3: Neuron Detail ────────────────────────────────────────────────────

function MiniHistogram({ histogram }) {
  if (!histogram) return null;
  const max = Math.max(...histogram.counts, 1);
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', height: 40, gap: 2, margin: '8px 0' }}>
      {histogram.counts.map((count, i) => (
        <div
          key={i}
          title={`[${histogram.bins[i]}, ${histogram.bins[i+1]}): ${count}`}
          style={{
            flex: 1, background: i === 4 || i === 5 ? '#4a3a8a' : '#2a2a50',
            height: `${Math.max(3, (count / max) * 100)}%`,
            borderRadius: '2px 2px 0 0', transition: 'height 0.3s'}}
        />
      ))}
    </div>
  );
}

function NeuronDetailPanel({ detail, model, onPatch, patchResult }) {
  if (!detail) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#444', fontSize: 14 }}>
        Select a neuron to inspect
      </div>
    );
  }

  const [patchVal, setPatchVal] = useState(3.5);

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: 20, background: 'var(--bg)' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#d0c0ff' }}>
            {model?.label} · L{detail.layer}N{detail.neuron_index}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 2 }}>
            {detail.activation_distribution} activation · sparsity {(detail.sparsity_score * 100).toFixed(0)}%
          </div>
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          {detail.circuit_memberships.map(c => <Chip key={c} label={c.replace('_', ' ')} color="#7c6af7" />)}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

        {/* Activation Histogram */}
        <div style={{ background: 'var(--bg-elev)', borderRadius: 10, padding: 16, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>Activation Histogram</div>
          <MiniHistogram histogram={detail.activation_histogram} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#555' }}>
            <span>μ={detail.activation_histogram.mean}</span>
            <span>σ={detail.activation_histogram.std}</span>
            <span>sparsity={(detail.activation_histogram.sparsity * 100).toFixed(0)}%</span>
          </div>
        </div>

        {/* Scores */}
        <div style={{ background: 'var(--bg-elev)', borderRadius: 10, padding: 16, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Importance Scores</div>
          {[
            { label: 'Attribution', value: detail.attribution_importance, color: '#7c6af7' },
            { label: 'Causal',      value: detail.causal_importance,      color: '#5cd4c4' },
            { label: 'Polysemanticity (lower is better)', value: detail.polysemanticity_score, color: '#e5a654' },
          ].map(({ label, value, color }) => (
            <div key={label} style={{ marginBottom: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#aaa' }}>
                <span>{label}</span><span style={{ color }}>{value.toFixed(3)}</span>
              </div>
              <ScoreBar value={value} color={color} />
            </div>
          ))}
        </div>

        {/* Top Activating Tokens */}
        <div style={{ background: 'var(--bg-elev)', borderRadius: 10, padding: 16, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Top Activating Tokens</div>
          {detail.top_activating_tokens.slice(0, 6).map((t, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <code style={{ background: '#1e1e3a', borderRadius: 3, padding: '1px 6px', fontSize: 11, color: '#c0b0ff', minWidth: 80 }}>
                {t.token}
              </code>
              <div style={{ flex: 1, position: 'relative' }}>
                <div style={{ height: 4, background: '#2a2a4a', borderRadius: 2 }}>
                  <div style={{ width: `${Math.min(100, (t.activation / 3.0) * 100)}%`, height: '100%', background: '#7c6af7', borderRadius: 2 }} />
                </div>
              </div>
              <span style={{ fontSize: 10, color: '#7c6af7', minWidth: 40, textAlign: 'right' }}>{t.activation?.toFixed(3)}</span>
            </div>
          ))}
        </div>

        {/* Negative Activations */}
        <div style={{ background: 'var(--bg-elev)', borderRadius: 10, padding: 16, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Negative Activations</div>
          {detail.negative_activating_tokens.map((t, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <code style={{ background: '#1e1e3a', borderRadius: 3, padding: '1px 6px', fontSize: 11, color: '#ff8080', minWidth: 80 }}>
                {t.token}
              </code>
              <span style={{ fontSize: 10, color: '#ff6060', marginLeft: 'auto' }}>{t.activation?.toFixed(3)}</span>
            </div>
          ))}
        </div>

        {/* SAE Feature Overlap */}
        <div style={{ background: 'var(--bg-elev)', borderRadius: 10, padding: 16, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>SAE Feature Overlap</div>
          {detail.sae_feature_overlap.map((f, i) => (
            <div key={i} style={{ marginBottom: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
                <span style={{ color: '#a0a0d0' }}>F{f.feature_id}: {f.description}</span>
                <span style={{ color: '#5cd4c4' }}>{(f.overlap_score * 100).toFixed(0)}%</span>
              </div>
              <ScoreBar value={f.overlap_score} color="#5cd4c4" />
            </div>
          ))}
        </div>

        {/* Connected Attention Heads */}
        <div style={{ background: 'var(--bg-elev)', borderRadius: 10, padding: 16, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Connected Attention Heads</div>
          {detail.connected_attention_heads.map((h, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <Chip label={`L${h.layer}H${h.head}`} color="#e5a654" />
              <ScoreBar value={h.importance} color="#e5a654" />
              <span style={{ fontSize: 10, color: '#e5a654' }}>{h.importance.toFixed(3)}</span>
            </div>
          ))}
        </div>

        {/* Patch Experiment */}
        <div style={{ background: 'var(--bg-elev)', borderRadius: 10, padding: 16, border: '1px solid var(--border)', gridColumn: '1 / -1' }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
            Activation Patch Experiment
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <div>
              <label style={{ fontSize: 11, color: '#aaa', display: 'block', marginBottom: 4 }}>Patch Value</label>
              <input
                id="patch-value-input"
                type="number" step="0.1"
                value={patchVal}
                onChange={e => setPatchVal(parseFloat(e.target.value) || 0)}
                style={{ width: 90, background: 'var(--bg)', border: '1px solid #3a3a6a', color: inherit, borderRadius: 6, padding: '6px 8px', fontSize: 13 }}
              />
            </div>
            <button
              id="patch-run-btn"
              onClick={() => onPatch(patchVal)}
              style={{
                background: 'linear-gradient(135deg, #7c6af7, #5cd4c4)',
                border: 'none', borderRadius: 8, padding: '8px 20px', fontSize: 13,
                fontWeight: 600, cursor: 'pointer', marginTop: 20}}
            >
              <svg width="12" height="12" viewBox="0 0 12 12" style={{ marginRight: 6, verticalAlign: 'middle' }}>
                <polygon points="2,0 12,6 2,12" fill="currentColor" />
              </svg>
              Run Patch
            </button>
            {patchResult && (
              <div style={{ display: 'flex', gap: 16, marginTop: 16, flexWrap: 'wrap' }}>
                <div style={{ background: 'var(--bg)', borderRadius: 8, padding: '8px 14px', textAlign: 'center' }}>
                  <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>Before</div>
                  <code style={{ color: '#c0b0ff', fontSize: 13 }}>{patchResult.top_token_before}</code>
                </div>
                <svg width="20" height="14" viewBox="0 0 20 14" style={{ color: '#7c6af7', alignSelf: 'center' }}>
                  <path d="M0 7 H16 M11 2 L16 7 L11 12" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <div style={{ background: 'var(--bg)', borderRadius: 8, padding: '8px 14px', textAlign: 'center' }}>
                  <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>After</div>
                  <code style={{ color: '#5cd4c4', fontSize: 13 }}>{patchResult.top_token_after}</code>
                </div>
                <div style={{ background: 'var(--bg)', borderRadius: 8, padding: '8px 14px', textAlign: 'center' }}>
                  <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>Δ logit</div>
                  <span style={{ color: patchResult.delta > 0 ? '#5cd4c4' : '#ff6060', fontSize: 13, fontWeight: 700 }}>
                    {patchResult.delta > 0 ? '+' : ''}{patchResult.delta?.toFixed(4)}
                  </span>
                </div>
              </div>
            )}
          </div>
          {detail.patch_experiment_history.length > 0 && (
            <div style={{ marginTop: 12, fontSize: 11, color: 'var(--text-dim)' }}>
              {detail.patch_experiment_history.length} prior experiment(s) on this neuron
            </div>
          )}
        </div>

        {/* Cross-Model Analogs */}
        <div style={{ background: 'var(--bg-elev)', borderRadius: 10, padding: 16, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Cross-Model Analogs</div>
          {detail.cross_model_analogs.map((a, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <span style={{ fontSize: 11, color: '#a0a0d0', minWidth: 80 }}>{a.model_id}</span>
              <span style={{ fontSize: 11, color: '#aaa' }}>L{a.layer}N{a.neuron_index}</span>
              <span style={{ marginLeft: 'auto', fontSize: 11, color: '#5cd4c4' }}>{(a.similarity * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>

        {/* Literature References */}
        <div style={{ background: 'var(--bg-elev)', borderRadius: 10, padding: 16, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Literature References</div>
          {detail.literature_references.map((ref, i) => (
            <div key={i} style={{ marginBottom: 8 }}>
              <div style={{ fontSize: 11, color: '#c0b0ff', marginBottom: 2 }}>{ref.title}</div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10 }}>
                <span style={{ color: 'var(--text-dim)' }}>arXiv:{ref.arxiv_id}</span>
                <span style={{ color: '#7c6af7' }}>relevance {(ref.relevance_score * 100).toFixed(0)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Main Neural Explorer View ─────────────────────────────────────────────────

export default function NeuralExplorerView({ api }) {
  const [selectedModel, setSelectedModel] = useState(MODELS[0]);
  const [treeData, setTreeData] = useState(null);
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [selectedNeuron, setSelectedNeuron] = useState(null);
  const [neuronDetail, setNeuronDetail] = useState(null);
  const [patchResult, setPatchResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const callApi = useCallback(async (path, params = {}) => {
    if (!api?.pythonCall) return null;
    try {
      return await api.pythonCall(path, params);
    } catch { return null; }
  }, [api]);

  // Load model tree when model changes
  useEffect(() => {
    setTreeData(null); setSelectedLayer(null); setSelectedNeuron(null); setNeuronDetail(null);
    if (!selectedModel) return;
    setLoading(true);
    callApi('api/v2/explorer/tree', { model_id: selectedModel.id }).then(data => {
      if (data) setTreeData(data);
      setLoading(false);
    });
  }, [selectedModel, callApi]);

  // Load neuron detail when neuron is selected
  useEffect(() => {
    if (!selectedNeuron) return;
    setPatchResult(null);
    callApi('api/v2/explorer/neuron', {
      model_id: selectedModel?.id || 'gpt2-small',
      layer: selectedNeuron.layer,
      neuron_index: selectedNeuron.neuron_index,
    }).then(data => { if (data) setNeuronDetail(data); });
  }, [selectedNeuron, selectedModel, callApi]);

  const handlePatch = useCallback(async (patchValue) => {
    if (!selectedNeuron) return;
    const result = await callApi('api/v2/explorer/patch_neuron', {
      model_id: selectedModel?.id || 'gpt2-small',
      layer: selectedNeuron.layer,
      neuron_index: selectedNeuron.neuron_index,
      patch_value: patchValue,
      prompt: 'The Eiffel Tower is in',
    });
    if (result) {
      setPatchResult(result);
      // Refresh neuron detail to show updated patch history
      const updated = await callApi('api/v2/explorer/neuron', {
        model_id: selectedModel?.id || 'gpt2-small',
        layer: selectedNeuron.layer,
        neuron_index: selectedNeuron.neuron_index,
      });
      if (updated) setNeuronDetail(updated);
    }
  }, [selectedNeuron, selectedModel, callApi]);

  return (
    <div
      id="neural-explorer-root"
      style={{ display: 'flex', height: '100%', background: 'var(--bg)', color: inherit, fontFamily: 'Inter, system-ui, sans-serif', overflow: 'hidden' }}
    >
      {/* Column 1 — Model tree */}
      <ModelTree
        selectedModel={selectedModel}
        onSelectModel={setSelectedModel}
        selectedLayer={selectedLayer}
        onSelectLayer={(layer) => { setSelectedLayer(layer); setSelectedNeuron(null); setNeuronDetail(null); }}
        selectedNeuron={selectedNeuron}
        onSelectNeuron={setSelectedNeuron}
        treeData={treeData}
      />

      {/* Column 2 — Layer detail */}
      <LayerDetail
        layer={selectedLayer}
        selectedNeuron={selectedNeuron}
        onSelectNeuron={setSelectedNeuron}
      />

      {/* Column 3 — Neuron detail */}
      {loading ? (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#555', fontSize: 14 }}>
          Loading model tree…
        </div>
      ) : (
        <NeuronDetailPanel
          detail={neuronDetail}
          model={selectedModel}
          onPatch={handlePatch}
          patchResult={patchResult}
        />
      )}
    </div>
  );
}

