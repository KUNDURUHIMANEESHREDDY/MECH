import React, { useState, useCallback, useEffect } from 'react';
import { NeuronUMAP } from './visualizations/neuron-umap/NeuronUMAP';
import { buildLayerNeuronPoints, idForLayerNeuron, parseNeuronId } from './visualizations/neuron-umap/data';
import { AttentionHeatmap } from './visualizations/panels/AttentionHeatmap';
import { colors, radii, spacing, typography } from '../design/tokens';

function fmtToken(tok) {
  if (!tok) return '';
  return String(tok).replaceAll('Ġ', '␣').replaceAll('Ċ', '⏎').trim();
}

function TierBadge({ tier }) {
  return (
    <span style={{
      background: colors.surfacePearl, color: colors.ink,
      borderRadius: 4, padding: '1px 8px', fontSize: 11, fontWeight: 700, marginLeft: 6}}>{tier}</span>
  );
}

function ScoreBar({ value, max = 1.0, color = colors.purple }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div style={{ background: 'var(--bg)', borderRadius: 4, height: 6, width: '100%', margin: '3px 0' }}>
      <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 4, transition: 'width 0.4s' }} />
    </div>
  );
}

function Chip({ label, color = colors.purple }) {
  return (
    <span style={{
      background: color + '22', color, border: `1px solid ${color}55`,
      borderRadius: 12, padding: '2px 10px', fontSize: 11, marginRight: 4, marginBottom: 4, display: 'inline-block'}}>{label}</span>
  );
}

function ModelTree({ selectedModel, onSelectModel, selectedLayer, onSelectLayer, selectedNeuron, onSelectNeuron, selectedHead, onSelectHead, treeData, models }) {
  const [expandedLayers, setExpandedLayers] = useState(new Set());

  const toggleLayer = (idx) => {
    setExpandedLayers(prev => {
      const next = new Set(prev);
      next.has(idx) ? next.delete(idx) : next.add(idx);
      return next;
    });
  };

  return (
    <div style={{ width: 220, minWidth: 180, background: colors.surfacePearl, borderRight: `1px solid ${colors.hairline}`, overflowY: 'auto', padding: '12px 0', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '0 12px 12px', borderBottom: `1px solid ${colors.hairline}` }}>
        <div style={{ fontSize: 10, color: colors.inkMuted48, marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 }}>Model</div>
        <select
          id="neural-explorer-model-select"
          value={selectedModel?.id || ''}
          onChange={e => onSelectModel(models.find(m => m.id === e.target.value))}
          style={{ width: '100%', background: 'var(--bg)', border: `1px solid ${colors.hairline}`, color: 'inherit', borderRadius: 6, padding: '6px 8px', fontSize: 12 }}
        >
          {models.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
        </select>
      </div>

      <div style={{ flex: 1, overflowY: 'auto' }}>
        {treeData?.layers?.map(layer => (
          <div key={layer.layer_index}>
            <button
              id={`layer-btn-${layer.layer_index}`}
              onClick={() => { onSelectLayer(layer); toggleLayer(layer.layer_index); }}
              style={{
                width: '100%', textAlign: 'left', padding: '7px 14px',
                background: selectedLayer?.layer_index === layer.layer_index ? colors.surfacePearl : 'transparent',
                border: 'none', color: colors.purpleBorder, fontSize: 12, cursor: 'pointer',
                borderLeft: selectedLayer?.layer_index === layer.layer_index ? `3px solid ${colors.purple}` : '3px solid transparent',
                display: 'flex', alignItems: 'center', gap: 6}}
            >
              <span style={{ display: 'inline-flex', color: colors.inkMuted48 }}>
                <svg width="8" height="8" viewBox="0 0 8 8" style={{ transform: expandedLayers.has(layer.layer_index) ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s' }}>
                  <polygon points="2,0 8,4 2,8" fill="currentColor" />
                </svg>
              </span>
              <span>Layer {layer.layer_index}</span>
            </button>

            {expandedLayers.has(layer.layer_index) && (
              <div style={{ paddingLeft: 24 }}>
                {layer.attention_heads_preview?.map(h => (
                  <button
                    key={h.head_index}
                    id={`tree-head-btn-${layer.layer_index}-${h.head_index}`}
                    title={`Head ${h.head_index} — Q L2: ${h.q_weight_l2?.toFixed(3)}`}
                    onClick={() => { onSelectLayer(layer); onSelectHead(layer.layer_index, h.head_index); }}
                    style={{
                      display: 'block', width: '100%', textAlign: 'left',
                      padding: '3px 8px', background: 'transparent', border: 'none',
                      color: selectedHead?.layer === layer.layer_index && selectedHead?.head_index === h.head_index ? colors.purple : colors.inkMuted48,
                      fontSize: 11, cursor: 'pointer'}}
                  >
                    H{h.head_index}
                    {h.top_token ? <span style={{ color: colors.purple, fontSize: 10, marginLeft: 4 }}>· {fmtToken(h.top_token)}</span> : null}
                  </button>
                ))}
                <div style={{ fontSize: 10, color: colors.inkMuted48, paddingLeft: 8, marginTop: 4 }}>
                  {layer.num_mlp_neurons} MLP neurons
                </div>
                {layer.mlp_neurons_preview?.length ? (
                  layer.mlp_neurons_preview.slice(0, 16).map(n => (
                    <button
                      key={n.neuron_index}
                      id={`neuron-btn-${layer.layer_index}-${n.neuron_index}`}
                      onClick={() => onSelectNeuron({ layer: layer.layer_index, neuron_index: n.neuron_index })}
                      style={{
                        display: 'block', width: '100%', textAlign: 'left',
                        padding: '3px 8px', background: 'transparent', border: 'none',
                        color: selectedNeuron?.layer === layer.layer_index && selectedNeuron?.neuron_index === n.neuron_index ? colors.purple : colors.inkMuted48,
                        fontSize: 11, cursor: 'pointer'}}
                    >
                      N{n.neuron_index}
                      {n.top_token ? <span style={{ color: colors.purple, fontSize: 10, marginLeft: 4 }}>· {fmtToken(n.top_token)}</span> : null}
                    </button>
                  ))
                ) : (
                  <div style={{ fontSize: 10, color: colors.inkMuted48, paddingLeft: 8 }}>Run a prompt to see active neurons</div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function LayerDetail({ layer, selectedHead, onSelectHead }) {
  if (!layer) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: colors.inkMuted48, fontSize: 14 }}>
        Select a layer to explore
      </div>
    );
  }
  return (
    <div style={{ width: 260, background: colors.surfacePearl, borderRight: `1px solid ${colors.hairline}`, overflowY: 'auto', padding: 16 }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: colors.purple, marginBottom: 12 }}>
        Layer {layer.layer_index}
        <span style={{ fontSize: 10, color: colors.inkMuted48, fontWeight: 400, marginLeft: 8 }}>d={layer.residual_stream_dim}</span>
      </div>

      <div style={{ fontSize: 10, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>
        Attention Heads ({layer.num_attention_heads}) · click to inspect
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 4, marginBottom: 16 }}>
        {layer.attention_heads_preview?.map(h => (
          <button
            key={h.head_index}
            id={`detail-head-btn-${layer.layer_index}-${h.head_index}`}
            title={`Head ${h.head_index}`}
            onClick={() => onSelectHead(layer.layer_index, h.head_index)}
            style={{
              padding: '4px 2px', borderRadius: 4, textAlign: 'center', fontSize: 10,
              background: selectedHead?.layer === layer.layer_index && selectedHead?.head_index === h.head_index ? colors.canvas : colors.canvas,
              border: selectedHead?.layer === layer.layer_index && selectedHead?.head_index === h.head_index ? `1px solid ${colors.primaryFocus}` : `1px solid ${colors.hairline}`,
              color: colors.inkMuted48,
              cursor: 'pointer'}}
          >
            H{h.head_index}
            {h.top_token ? <div style={{ fontSize: 9, color: colors.purple, marginTop: 2 }}>{fmtToken(h.top_token)}</div> : null}
          </button>
        ))}
      </div>

      <div style={{ fontSize: 10, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>
        Top Active Neurons
      </div>
      {layer.mlp_neurons_preview?.length ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 3 }}>
          {layer.mlp_neurons_preview?.slice(0, 32).map(n => (
            <div
              key={n.neuron_index}
              title={n.label}
              style={{
                padding: '4px 2px', borderRadius: 4, textAlign: 'center', fontSize: 10,
                background: colors.canvas,
                border: `1px solid ${colors.hairline}`,
                color: colors.inkMuted48,
                cursor: 'default'}}
            >
              {n.neuron_index}
              {n.top_token ? <div style={{ fontSize: 9, color: colors.purple, marginTop: 2 }}>{fmtToken(n.top_token)}</div> : null}
            </div>
          ))}
        </div>
      ) : (
        <div style={{ fontSize: 11, color: colors.inkMuted48 }}>Run a prompt to populate top active neurons.</div>
      )}
    </div>
  );
}

function MiniHistogram({ histogram }) {
  if (!histogram || !histogram.counts) return null;
  const max = Math.max(...histogram.counts, 1);
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', height: 40, gap: 2, margin: '8px 0' }}>
      {histogram.counts.map((count, i) => (
        <div
          key={i}
          title={`[${histogram.bins[i]}, ${histogram.bins[i+1]}): ${count}`}
          style={{
            flex: 1, background: i === 4 || i === 5 ? `${colors.purple}66` : `${colors.canvas}`,
            height: `${Math.max(3, (count / max) * 100)}%`,
            borderRadius: '2px 2px 0 0', transition: 'height 0.3s'}}
        />
      ))}
    </div>
  );
}

function WeightBar({ dim, weight, maxAbs }) {
  const pct = Math.min(100, (Math.abs(weight) / maxAbs) * 100);
  const color = weight >= 0 ? colors.purple : colors.danger;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, marginBottom: 2 }}>
      <span style={{ color: colors.inkMuted48, minWidth: 40, fontSize: 9 }}>D{dim}</span>
      <div style={{ flex: 1, height: 6, background: colors.hairline, borderRadius: 2, overflow: 'hidden' }}>
        <div style={{
          width: `${pct}%`,
          height: '100%',
          background: color,
          borderRadius: 2,
          transform: 'translateX(0)',
        }} />
      </div>
      <span style={{ color, minWidth: 50, textAlign: 'right' }}>{weight.toFixed(4)}</span>
    </div>
  );
}

function NeuronDetailPanel({ detail, model, onPatch, patchResult }) {
  const [patchVal, setPatchVal] = useState(3.0);

  if (!detail) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: colors.inkMuted48, fontSize: 14 }}>
        Select a neuron to inspect
      </div>
    );
  }

  const perTok = detail.per_token_activations || [];
  const activatesOn = perTok.length > 0
    ? perTok.reduce((best, ta) => (ta.activation > best.activation ? ta : best), perTok[0])
    : null;

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: 20, background: 'var(--bg)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 700, color: colors.purpleBorder }}>
            {model?.label || 'GPT-2'} · {detail.id || `L${detail.layer}N${detail.neuron_index}`}
          </div>
          <div style={{ fontSize: 11, color: colors.inkMuted48, marginTop: 2 }}>
            {detail.description}
          </div>
          {activatesOn && (
            <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 10, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1 }}>
                Activates on
              </span>
              <code style={{
                background: `${colors.purple}22`, border: `1px solid ${colors.purple}55`, color: colors.purpleBorder,
                borderRadius: 6, padding: '2px 10px', fontSize: 14, fontWeight: 600}}>
                {fmtToken(activatesOn.token)}
              </code>
            </div>
          )}
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <Chip label={detail.component} color={colors.purple} />
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

        <div style={{ background: 'var(--bg-elev)', borderRadius: 10, padding: 16, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: colors.inkMuted48, marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>Weight Summary</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 11 }}>
            <span style={{ color: colors.inkMuted48 }}>Bias</span>
            <span style={{ color: colors.bodyMuted }}>{detail.bias !== null ? detail.bias.toFixed(4) : 'N/A'}</span>
            <span style={{ color: colors.inkMuted48 }}>In Weight L2</span>
            <span style={{ color: colors.bodyMuted }}>{detail.in_weight_l2 !== null ? detail.in_weight_l2.toFixed(4) : 'N/A'}</span>
            <span style={{ color: colors.inkMuted48 }}>Out Weight L2</span>
            <span style={{ color: colors.bodyMuted }}>{detail.out_weight_l2 !== null ? detail.out_weight_l2.toFixed(4) : 'N/A'}</span>
            <span style={{ color: colors.inkMuted48 }}>In Mean</span>
            <span style={{ color: colors.bodyMuted }}>{detail.in_weight_stats?.mean !== undefined ? detail.in_weight_stats.mean.toFixed(4) : 'N/A'}</span>
            <span style={{ color: colors.inkMuted48 }}>Out Mean</span>
            <span style={{ color: colors.bodyMuted }}>{detail.out_weight_stats?.mean !== undefined ? detail.out_weight_stats.mean.toFixed(4) : 'N/A'}</span>
          </div>
        </div>

        <div style={{ background: 'var(--bg-elev)', borderRadius: 10, padding: 16, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: colors.inkMuted48, marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>Activation Stats</div>
          {detail.activation_stats ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 11 }}>
              <span style={{ color: colors.inkMuted48 }}>Mean</span><span style={{ color: colors.bodyMuted }}>{detail.activation_stats.mean.toFixed(4)}</span>
              <span style={{ color: colors.inkMuted48 }}>Std</span><span style={{ color: colors.bodyMuted }}>{detail.activation_stats.std.toFixed(4)}</span>
              <span style={{ color: colors.inkMuted48 }}>Min</span><span style={{ color: colors.bodyMuted }}>{detail.activation_stats.min.toFixed(4)}</span>
              <span style={{ color: colors.inkMuted48 }}>Max</span><span style={{ color: colors.bodyMuted }}>{detail.activation_stats.max.toFixed(4)}</span>
            </div>
          ) : (
            <span style={{ color: colors.inkMuted48, fontSize: 11 }}>No activations — run a prompt first</span>
          )}
          <MiniHistogram histogram={detail.activation_histogram} />
        </div>

        <div style={{ background: 'var(--bg-elev)', borderRadius: 10, padding: 16, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: colors.inkMuted48, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Top Input Weights</div>
          {detail.top_input_weights_positive?.slice(0, 8).map((w) => (
            <WeightBar key={w.dim} dim={w.dim} weight={w.weight} maxAbs={Math.max(...detail.top_input_weights_positive.map(x => Math.abs(x.weight)), ...detail.top_input_weights_negative.map(x => Math.abs(x.weight)))} />
          ))}
          {detail.top_input_weights_negative?.slice(0, 4).map((w) => (
            <WeightBar key={'neg-' + w.dim} dim={w.dim} weight={w.weight} maxAbs={Math.max(...detail.top_input_weights_positive.map(x => Math.abs(x.weight)), ...detail.top_input_weights_negative.map(x => Math.abs(x.weight)))} />
          ))}
        </div>

        <div style={{ background: 'var(--bg-elev)', borderRadius: 10, padding: 16, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: colors.inkMuted48, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Per-Token Activations</div>
          {detail.per_token_activations && detail.per_token_activations.length > 0 ? (
            detail.per_token_activations.slice(0, 12).map((ta) => (
              <div key={ta.token_index} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <code style={{ background: colors.surfacePearl, borderRadius: 3, padding: '1px 6px', fontSize: 11, color: colors.purpleBorder, minWidth: 80 }}>
                  {ta.token}
                </code>
                <div style={{ flex: 1, height: 4, background: colors.hairline, borderRadius: 2, position: 'relative', overflow: 'visible' }}>
                  <div style={{
                    position: 'absolute',
                    left: ta.activation >= 0 ? '50%' : '50%',
                    width: `${Math.min(50, Math.abs(ta.activation) / 5 * 100)}%`,
                    height: '100%',
                    background: ta.activation >= 0 ? colors.primary : colors.danger,
                    borderRadius: 2,
                    transform: ta.activation >= 0 ? 'translateX(0)' : 'translateX(-100%)',
                  }} />
                </div>
                <span style={{ fontSize: 10, color: colors.inkMuted48, minWidth: 60, textAlign: 'right' }}>
                  {ta.activation.toFixed(4)}
                </span>
                {ta.pre_activation !== null && (
                  <span style={{ fontSize: 9, color: colors.inkMuted48 }}>pre: {ta.pre_activation.toFixed(3)}</span>
                )}
              </div>
            ))
          ) : (
            <span style={{ color: colors.inkMuted48, fontSize: 11 }}>No cached activations.</span>
          )}
        </div>

        <div style={{ background: 'var(--bg-elev)', borderRadius: 10, padding: 16, border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: colors.inkMuted48, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Nearest Neurons (by weights)</div>
          {detail.nearest_neurons && detail.nearest_neurons.length > 0 ? (
            detail.nearest_neurons.map((n) => (
              <div key={n.neuron_index} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
                <span style={{ color: colors.inkMuted48 }}>L{n.layer}N{n.neuron_index}</span>
                <span style={{ color: colors.primary }}>{(n.similarity * 100).toFixed(1)}%</span>
              </div>
            ))
          ) : (
            <span style={{ color: colors.inkMuted48, fontSize: 11 }}>No nearest neighbors.</span>
          )}
        </div>

        <div style={{ background: 'var(--bg-elev)', borderRadius: 10, padding: 16, border: '1px solid var(--border)', gridColumn: '1 / -1' }}>
          <div style={{ fontSize: 11, color: colors.inkMuted48, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
            Activation Patch Experiment
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <div>
              <label style={{ fontSize: 11, color: colors.inkMuted48, display: 'block', marginBottom: 4 }}>Patch Value</label>
              <input
                id="patch-value-input"
                type="number" step="0.1"
                value={patchVal}
                onChange={e => setPatchVal(parseFloat(e.target.value) || 0)}
                style={{ width: 90, background: 'var(--bg)', border: `1px solid ${colors.hairline}`, color: 'inherit', borderRadius: 6, padding: '6px 8px', fontSize: 13 }}
              />
            </div>
            <button
              id="patch-run-btn"
              onClick={() => onPatch(patchVal)}
              style={{
background: `linear-gradient(135deg, ${colors.purple}, ${colors.primary})`,
                border: 'none', borderRadius: 8, padding: '8px 20px', fontSize: 13,
                fontWeight: 600, color: colors.onPrimary, cursor: 'pointer', marginTop: 20}}
            >
              <svg width="12" height="12" viewBox="0 0 12 12" style={{ marginRight: 6, verticalAlign: 'middle' }}>
                <polygon points="2,0 12,6 2,12" fill="currentColor" />
              </svg>
              Run Patch
            </button>
            {patchResult && (
              <div style={{ display: 'flex', gap: 16, marginTop: 16, flexWrap: 'wrap' }}>
                <div style={{ background: 'var(--bg)', borderRadius: 8, padding: '8px 14px', textAlign: 'center' }}>
                  <div style={{ fontSize: 10, color: colors.inkMuted48 }}>Before</div>
                  <code style={{ color: colors.purpleBorder, fontSize: 13 }}>{patchResult.top_token_before}</code>
                </div>
                <svg width="20" height="14" viewBox="0 0 20 14" style={{ color: colors.purple, alignSelf: 'center' }}>
                  <path d="M0 7 H16 M11 2 L16 7 L11 12" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <div style={{ background: 'var(--bg)', borderRadius: 8, padding: '8px 14px', textAlign: 'center' }}>
                  <div style={{ fontSize: 10, color: colors.inkMuted48 }}>After</div>
                  <code style={{ color: colors.primary, fontSize: 13 }}>{patchResult.top_token_after}</code>
                </div>
                <div style={{ background: 'var(--bg)', borderRadius: 8, padding: '8px 14px', textAlign: 'center' }}>
                  <div style={{ fontSize: 10, color: colors.inkMuted48 }}>Δ logit</div>
                  <span style={{ color: patchResult.delta > 0 ? colors.primary : colors.danger, fontSize: 13, fontWeight: 700 }}>
                    {patchResult.delta > 0 ? '+' : ''}{patchResult.delta?.toFixed(4)}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function HeadDetailPanel({ head, detail, loading, error }) {
  const [hovered, setHovered] = useState(null);
  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: 20, background: 'var(--bg)' }}>
      <div style={{ fontSize: 18, fontWeight: 700, color: colors.purpleBorder, marginBottom: 4 }}>
        GPT-2 · L{head.layer}H{head.head_index}
      </div>
      <div style={{ fontSize: 11, color: colors.inkMuted48, marginBottom: 16 }}>
        Attention pattern — rows = query tokens, cols = key tokens
      </div>
      {loading && <div style={{ color: colors.inkMuted48, fontSize: 13 }}>Loading attention pattern…</div>}
      {error && (
        <div style={{
          background: `${colors.danger}18`, border: `1px solid ${colors.danger}55`, color: colors.danger,
          borderRadius: 8, padding: '12px 16px', fontSize: 12, marginBottom: 16, display: 'inline-block'}}
        >
          {error}
        </div>
      )}
      {detail && (
        <div style={{ background: 'var(--bg-elev)', borderRadius: 10, padding: 16, border: '1px solid var(--border)', display: 'inline-block' }}>
          <AttentionHeatmap
            matrix={detail.matrix}
            tokens={detail.str_tokens || []}
            hoveredToken={hovered}
            onHoverToken={setHovered}
          />
        </div>
      )}
    </div>
  );
}

export default function NeuralExplorerView({ api }) {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [treeData, setTreeData] = useState(null);
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [selectedNeuron, setSelectedNeuron] = useState(null);
  const [neuronDetail, setNeuronDetail] = useState(null);
  const [patchResult, setPatchResult] = useState(null);
  const [selectedHead, setSelectedHead] = useState(null);
  const [headDetail, setHeadDetail] = useState(null);
  const [headError, setHeadError] = useState(null);
  const [headLoading, setHeadLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [prompt, setPrompt] = useState('The capital of France is');
  const [promptRunning, setPromptRunning] = useState(false);
  const [promptMsg, setPromptMsg] = useState(null);
  const [hasActivations, setHasActivations] = useState(false);
  const [strTokens, setStrTokens] = useState([]);

  const callApi = useCallback(async (path, params = {}) => {
    if (!api?.pythonCall) return null;
    try {
      return await api.pythonCall(path, params);
    } catch { return null; }
  }, [api]);

  useEffect(() => {
    const loadModels = async () => {
      setLoading(true);
      try {
        if (api && api.gpt2Architecture) {
          const arch = await api.gpt2Architecture();
          if (arch.status === 'ok') {
            const model = {
              id: 'gpt2-small',
              label: arch.model_name || 'GPT-2',
              family: arch.model_type || 'gpt2',
              layers: arch.n_layers,
              heads: arch.n_heads,
              d_model: arch.d_model,
              d_mlp: arch.d_mlp,
            };
            setModels([model]);
            setSelectedModel(model);
          }
        }
      } catch { } finally {
        setLoading(false);
      }
    };
    loadModels();
  }, []);

  const loadTree = useCallback(async () => {
    if (!selectedModel) return;
    setLoading(true);
    const layers = [];
    let anyActivations = false;
    for (let li = 0; li < selectedModel.layers; li++) {
      const res = await callApi('gpt2/layer', { layer: li });
      if (res && res.status === 'ok') {
        const preview = res.top_active_neurons || [];
        if (preview.length > 0) anyActivations = true;
        if (res.str_tokens) setStrTokens(res.str_tokens);
        layers.push({
          layer_index: res.layer,
          label: res.path || `blocks.${li}`,
          num_attention_heads: res.num_attention_heads,
          num_mlp_neurons: res.num_mlp_neurons,
          residual_stream_dim: res.residual_stream_dim,
          n_params: res.n_params,
          attention_heads_preview: res.attention_heads || [],
          mlp_neurons_preview: preview,
          known_circuits: [],
        });
      }
    }
    setTreeData({ layers });
    setHasActivations(anyActivations);
    if (layers.length > 0) {
      setSelectedLayer(prev => layers.find(l => l.layer_index === prev?.layer_index) || layers[0]);
    }
    setLoading(false);
  }, [selectedModel, callApi]);

  useEffect(() => {
    setTreeData(null); setSelectedLayer(null); setSelectedNeuron(null); setNeuronDetail(null);
    setHasActivations(false); setStrTokens([]);
    if (!selectedModel) return;
    loadTree();
  }, [selectedModel, callApi, loadTree]);

  useEffect(() => {
    if (!selectedNeuron) return;
    setPatchResult(null);
    callApi('gpt2/neuron', {
      layer: selectedNeuron.layer,
      neuron_index: selectedNeuron.neuron_index,
      component: 'mlp',
      top_k_weights: 16,
    }).then(data => { if (data) setNeuronDetail(data); });
  }, [selectedNeuron, callApi]);

  useEffect(() => {
    if (!selectedHead) return;
    setHeadLoading(true); setHeadError(null); setHeadDetail(null);
    callApi('gpt2/attention_head', {
      layer: selectedHead.layer,
      head: selectedHead.head_index,
    }).then(data => {
      setHeadLoading(false);
      if (!data) { setHeadError('No response from the Python backend.'); return; }
      if (data.status === 'ok') setHeadDetail(data);
      else setHeadError(data.error || 'Failed to load attention pattern.');
    });
  }, [selectedHead, callApi]);

  const handleSelectNeuron = useCallback((sel) => {
    setSelectedNeuron(sel);
    setSelectedHead(null); setHeadDetail(null); setHeadError(null);
  }, []);

  const handleSelectHead = useCallback((layerIndex, headIndex) => {
    setSelectedHead({ layer: layerIndex, head_index: headIndex });
    setSelectedNeuron(null); setNeuronDetail(null); setPatchResult(null);
  }, []);

  const handlePatch = useCallback(async (patchValue) => {
    if (!selectedNeuron) return;
    const result = await callApi('gpt2/patch_neuron', {
      layer: selectedNeuron.layer,
      neuron_index: selectedNeuron.neuron_index,
      patch_value: patchValue,
      prompt: 'The Eiffel Tower is in',
    });
    if (result) {
      setPatchResult(result);
      const updated = await callApi('gpt2/neuron', {
        layer: selectedNeuron.layer,
        neuron_index: selectedNeuron.neuron_index,
        component: 'mlp',
        top_k_weights: 16,
      });
      if (updated) setNeuronDetail(updated);
    }
  }, [selectedNeuron, callApi]);

  const handleRunPrompt = useCallback(async () => {
    const text = prompt.trim();
    if (!api?.gpt2RunPrompt || !text || promptRunning) return;
    setPromptRunning(true);
    setPromptMsg(null);
    try {
      const res = await api.gpt2RunPrompt(text);
      if (res && res.status === 'ok') {
        setPromptMsg({
          ok: true,
          text: `Prompt ran — ${(res.str_tokens || []).length} tokens. Reloading activation map…`,
        });
        await loadTree();
        setPromptMsg(prev => prev ? { ...prev, text: `Prompt ran — activation map populated.` } : prev);
      } else {
        setPromptMsg({ ok: false, text: 'Prompt failed — check the Python backend status.' });
      }
    } catch (e) {
      setPromptMsg({ ok: false, text: `Error running prompt: ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      setPromptRunning(false);
    }
  }, [api, prompt, promptRunning, loadTree]);

  const selTokIdx = (() => {
    if (!selectedNeuron || !treeData) return null;
    const layer = treeData.layers.find(l => l.layer_index === selectedNeuron.layer);
    const n = layer?.mlp_neurons_preview?.find(x => x.neuron_index === selectedNeuron.neuron_index);
    return n && typeof n.top_token_index === 'number' ? n.top_token_index : null;
  })();

  return (
    <div
      id="neural-explorer-root"
      style={{
        display: 'flex', flexDirection: 'column', height: '100%',
        background: 'var(--bg)', color: 'inherit',
        fontFamily: "'Inter', system-ui, sans-serif", overflow: 'hidden',
        '--bg': colors.canvasParchment,
        '--bg-elev': colors.canvas,
        '--border': colors.hairline,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 16px', background: colors.surfacePearl, borderBottom: `1px solid ${colors.hairline}`, flexShrink: 0 }}>
        <span style={{ fontSize: 10, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1 }}>Prompt</span>
        <input
          id="ne-prompt-input"
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') handleRunPrompt(); }}
          placeholder="Run a prompt to populate activations…"
          style={{ flex: 1, minWidth: 0, background: 'var(--bg)', border: `1px solid ${colors.hairline}`, color: 'inherit', borderRadius: 6, padding: '6px 10px', fontSize: 12 }}
        />
        <button
          id="ne-run-prompt-btn"
          onClick={handleRunPrompt}
          disabled={promptRunning || !selectedModel}
          style={{
            background: `linear-gradient(135deg, ${colors.purple}, ${colors.primary})`,
            border: 'none', borderRadius: 8, padding: '7px 18px', fontSize: 12,
            fontWeight: 600, color: colors.onPrimary, cursor: promptRunning ? 'default' : 'pointer',
            opacity: promptRunning ? 0.65 : 1}}
        >
          {promptRunning ? 'Running…' : 'Run Prompt'}
        </button>
        <button
          id="ne-refresh-btn"
          onClick={() => loadTree()}
          title="Reload layer data"
          style={{
            background: 'transparent', border: `1px solid ${colors.hairline}`, borderRadius: 8,
            padding: '7px 14px', fontSize: 12, color: colors.purpleBorder, cursor: 'pointer'}}
        >
          Refresh
        </button>
      </div>

      {promptMsg && (
        <div style={{
          padding: '6px 16px', fontSize: 11, flexShrink: 0,
          color: promptMsg.ok ? colors.primary : colors.danger,
          background: promptMsg.ok ? `${colors.success}18` : `${colors.danger}18`,
          borderBottom: `1px solid ${colors.hairline}`}}
        >
          {promptMsg.text}
        </div>
      )}

      {strTokens.length > 0 && (
        <div
          data-testid="ne-token-strip"
          style={{
          display: 'flex', flexWrap: 'wrap', gap: 4, padding: '6px 16px', flexShrink: 0,
          background: colors.surfacePearl, borderBottom: `1px solid ${colors.hairline}`}}
        >
          {strTokens.map((t, i) => (
            <span
              key={i}
              data-testid={`ne-token-chip-${i}`}
              style={{
              fontSize: 10, padding: '2px 6px', borderRadius: 4,
              background: i === selTokIdx ? `${colors.purple}33` : colors.canvas,
              border: i === selTokIdx ? `1px solid ${colors.purple}` : `1px solid ${colors.hairline}`,
              color: i === selTokIdx ? colors.purpleBorder : colors.inkMuted48,
              fontWeight: i === selTokIdx ? 600 : 400}}>
              {fmtToken(t)}
            </span>
          ))}
        </div>
      )}

      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
      <ModelTree
        selectedModel={selectedModel}
        onSelectModel={setSelectedModel}
        selectedLayer={selectedLayer}
        onSelectLayer={(layer) => { setSelectedLayer(layer); setSelectedNeuron(null); setNeuronDetail(null); setSelectedHead(null); setHeadDetail(null); setHeadError(null); }}
        selectedNeuron={selectedNeuron}
        onSelectNeuron={handleSelectNeuron}
        selectedHead={selectedHead}
        onSelectHead={handleSelectHead}
        treeData={treeData}
        models={models}
      />

      <LayerDetail
        layer={selectedLayer}
        selectedHead={selectedHead}
        onSelectHead={handleSelectHead}
      />

      {selectedLayer && selectedLayer.mlp_neurons_preview && (
        <div style={{ width: 430, background: colors.surfacePearl, borderRight: `1px solid ${colors.hairline}`, overflowY: 'auto', padding: 16, flexShrink: 0 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: colors.purple, marginBottom: 12 }}>
            Neuron Map — Layer {selectedLayer.layer_index}
          </div>
          {!hasActivations && (
            <div style={{ fontSize: 11, color: colors.inkMuted48, marginBottom: 10 }}>
              No activations yet — run a prompt above to populate the map.
            </div>
          )}
          <NeuronUMAP
            points={buildLayerNeuronPoints(selectedLayer.mlp_neurons_preview, selectedLayer.layer_index)}
            selectedId={selectedNeuron && selectedNeuron.layer === selectedLayer.layer_index
              ? idForLayerNeuron(selectedNeuron.layer, selectedNeuron.neuron_index)
              : null}
            onSelectNeuron={(id) => {
              const parsed = parseNeuronId(id);
              handleSelectNeuron(parsed ? { layer: parsed.layer, neuron_index: parsed.neuron } : null);
            }}
            height={460}
          />
        </div>
      )}

      {loading ? (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: colors.inkMuted48, fontSize: 14 }}>
          Loading model tree…
        </div>
      ) : selectedHead ? (
        <HeadDetailPanel
          head={selectedHead}
          detail={headDetail}
          loading={headLoading}
          error={headError}
        />
      ) : (
        <NeuronDetailPanel
          detail={neuronDetail}
          model={selectedModel}
          onPatch={handlePatch}
          patchResult={patchResult}
        />
      )}
      </div>
    </div>
  );
}
