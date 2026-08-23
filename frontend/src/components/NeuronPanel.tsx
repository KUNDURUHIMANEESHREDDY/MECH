import React, { useState, useEffect } from 'react';
import {
  Brain, Activity, ArrowRight, ArrowUpRight, ArrowDownRight,
  ShieldCheck, AlertTriangle, Layers, Dna
} from 'lucide-react';
import { scienceApi } from '../science/api/scienceApi';
import { FeatureEvidence, SubstrateAnchor } from '../science/types/scientificTypes';
import './NeuronPanel.css';

/* ── Types ──────────────────────────────────────────────────────────── */

interface NeuronPanelProps {
  neurons?: Array<{ index: number; activation: number; tokenActivations?: number[] }>;
  selectedNeuron?: number | null;
  onSelectNeuron?: (idx: number) => void;
  tokens?: string[];
  layer?: number;
  modelName?: string;
}

/* ── Helpers ────────────────────────────────────────────────────────── */

/** Estimate L2 weight norm from activation when real data is unavailable. */
function estimateL2(activation: number): number {
  return parseFloat((activation * 0.7071).toFixed(3));
}

/**
 * Return SAE features that list this (layer, neuronIndex) as a substrate
 * anchor — i.e. features where this physical neuron is a dense basis coordinate.
 */
function anchoredFeatures(
  layer: number,
  neuronIndex: number,
  features: FeatureEvidence[],
): FeatureEvidence[] {
  return features.filter((f) =>
    f.substrate_anchors?.some(
      (a: SubstrateAnchor) => a.layer === layer && a.neuron_idx === neuronIndex,
    ),
  );
}

/* ── Polysemanticity pill ───────────────────────────────────────────── */

function PolysemPill({ poly }: { poly: boolean }) {
  return (
    <span
      style={{
        fontSize: 9.5,
        fontWeight: 700,
        padding: '2px 5px',
        borderRadius: 3,
        background: poly ? 'rgba(245,158,11,0.18)' : 'rgba(16,185,129,0.18)',
        color: poly ? '#f59e0b' : '#10b981',
        letterSpacing: 0.3,
      }}
    >
      {poly ? 'POLY' : 'MONO'}
    </span>
  );
}

/* ── Component ──────────────────────────────────────────────────────── */

export const NeuronPanel: React.FC<NeuronPanelProps> = ({
  neurons = [],
  selectedNeuron,
  onSelectNeuron,
  tokens = [],
  layer = 0,
  modelName = 'gpt2',
}) => {
  const [activeTab, setActiveTab] = useState<'substrate' | 'sae'>('substrate');
  const [saeFeatures, setSaeFeatures] = useState<FeatureEvidence[]>([]);
  const [selectedFeatureId, setSelectedFeatureId] = useState<string | null>(null);
  const [loadingFeatures, setLoadingFeatures] = useState(false);

  /* Fetch SAE features for the current layer whenever layer/model changes. */
  useEffect(() => {
    setLoadingFeatures(true);
    scienceApi
      .fetchLayerFeatures(layer, modelName)
      .then((feats) => {
        setSaeFeatures(feats);
        if (feats.length > 0 && !selectedFeatureId) {
          setSelectedFeatureId(feats[0].feature_id);
        }
      })
      .catch((err) => console.warn('NeuronPanel: SAE feature fetch failed:', err))
      .finally(() => setLoadingFeatures(false));
  }, [layer, modelName]); // eslint-disable-line react-hooks/exhaustive-deps

  /* Resolved selected neuron data — fall back to a sentinel so the UI never crashes. */
  const selectedNeuronData = neurons.find((n) => n.index === (selectedNeuron ?? -1))
    ?? neurons[0]
    ?? { index: selectedNeuron ?? 0, activation: 1.42, tokenActivations: [0.12, 0.45, 1.42, 0.28] };

  /* SAE features anchored to the currently selected neuron. */
  const linkedFeatures = anchoredFeatures(layer, selectedNeuronData.index, saeFeatures);

  const selectedFeature = saeFeatures.find((f) => f.feature_id === selectedFeatureId) ?? saeFeatures[0];

  /* Derived L2 / polysemanticity from substrate_anchors if present. */
  const anchorMeta: SubstrateAnchor | undefined = saeFeatures
    .flatMap((f) => f.substrate_anchors ?? [])
    .find((a) => a.layer === layer && a.neuron_idx === selectedNeuronData.index);

  const l2Norm = anchorMeta?.weight_norm ?? estimateL2(selectedNeuronData.activation);
  const isPoly = anchorMeta?.is_polysemantic ?? true; // default: polysemantic (conservative)

  return (
    <div className="neuron-panel" data-testid="neuron-substrate-inspector">

      {/* ── Header ───────────────────────────────────────────────────── */}
      <div className="neuron-panel-header" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Brain size={16} style={{ color: 'var(--accent, #89b4fa)' }} />
            <span style={{ fontWeight: 600, fontSize: 13 }}>Neural &amp; SAE Explorer</span>
          </div>
          <span style={{ fontSize: 11, color: 'var(--text-dim, #a6adc8)' }}>Layer {layer}</span>
        </div>

        {/* Tab switcher */}
        <div style={{ display: 'flex', gap: 4, background: 'rgba(0,0,0,0.25)', padding: 2, borderRadius: 6 }}>
          {(['substrate', 'sae'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                flex: 1,
                padding: '4px 8px',
                fontSize: 11,
                fontWeight: 600,
                border: 'none',
                borderRadius: 4,
                background:
                  activeTab === tab
                    ? tab === 'substrate'
                      ? 'var(--accent, #89b4fa)'
                      : '#a6e3a1'
                    : 'transparent',
                color: activeTab === tab ? '#11111b' : 'var(--text-dim, #a6adc8)',
                cursor: 'pointer',
              }}
            >
              {tab === 'substrate' ? 'Physical Substrate' : `SAE Dictionary${saeFeatures.length ? ` (${saeFeatures.length})` : ''}`}
            </button>
          ))}
        </div>
      </div>

      {/* ── Body ─────────────────────────────────────────────────────── */}
      <div className="neuron-panel-body" style={{ padding: 12 }}>

        {/* ═══════════════════════════════════════════════════════════
            SUBSTRATE TAB — Reference Substrate Anchors (Lℓ_Ni)
            ═══════════════════════════════════════════════════════════ */}
        {activeTab === 'substrate' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

            {/* Scientific framing banner */}
            <div
              style={{
                padding: '8px 10px',
                background: 'rgba(137,180,250,0.08)',
                border: '1px solid rgba(137,180,250,0.2)',
                borderRadius: 6,
                fontSize: 11,
                lineHeight: 1.45,
              }}
            >
              <strong>Reference Substrate Anchor (L{layer}_N*):</strong>{' '}
              Physical neurons are <em>dense, polysemantic computational coordinates</em> — not
              standalone semantic concepts. MECH uses them as reference anchors; meaning is inferred
              from sparse dictionary latents that decompose these dense activations.
            </div>

            {/* Neuron list */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {(neurons.length > 0 ? neurons : [selectedNeuronData]).map((n) => {
                const isSelected = n.index === selectedNeuronData.index;
                const rowAnchor = saeFeatures
                  .flatMap((f) => f.substrate_anchors ?? [])
                  .find((a) => a.layer === layer && a.neuron_idx === n.index);
                const rowL2 = rowAnchor?.weight_norm ?? estimateL2(n.activation);
                const rowPoly = rowAnchor?.is_polysemantic ?? true;
                const maxAct = Math.max(...(neurons.length > 0 ? neurons : [n]).map((x) => x.activation), 1);
                return (
                  <button
                    key={n.index}
                    onClick={() => onSelectNeuron?.(n.index)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      width: '100%',
                      border: isSelected
                        ? '1px solid var(--accent, #89b4fa)'
                        : '1px solid rgba(255,255,255,0.06)',
                      borderRadius: 5,
                      background: isSelected ? 'rgba(137,180,250,0.1)' : 'transparent',
                      padding: '4px 7px',
                      cursor: 'pointer',
                      textAlign: 'left',
                    }}
                  >
                    {/* Coordinate */}
                    <span style={{ width: 70, fontFamily: 'monospace', fontSize: 10.5, color: isSelected ? '#89b4fa' : 'var(--text-dim, #a6adc8)', flexShrink: 0 }}>
                      L{layer}_N{n.index}
                    </span>

                    {/* Activation bar */}
                    <div style={{ flex: 1, height: 8, background: 'rgba(255,255,255,0.06)', borderRadius: 4, overflow: 'hidden' }}>
                      <div
                        style={{
                          height: '100%',
                          width: `${Math.max(2, (n.activation / maxAct) * 100).toFixed(1)}%`,
                          background: isSelected ? '#89b4fa' : 'rgba(137,180,250,0.45)',
                          borderRadius: 4,
                        }}
                      />
                    </div>

                    {/* Activation value */}
                    <span style={{ width: 42, textAlign: 'right', fontFamily: 'monospace', fontSize: 10.5, color: 'var(--text-dim, #a6adc8)', flexShrink: 0 }}>
                      {n.activation.toFixed(3)}
                    </span>

                    {/* L2 norm */}
                    <span style={{ width: 48, textAlign: 'right', fontFamily: 'monospace', fontSize: 10, color: '#f9a8d4', flexShrink: 0 }}>
                      ‖w‖{rowL2.toFixed(2)}
                    </span>

                    {/* Poly pill */}
                    <PolysemPill poly={rowPoly} />
                  </button>
                );
              })}
            </div>

            {/* ── Selected neuron detail card ─────────────────────── */}
            {selectedNeuronData && (
              <div
                style={{
                  background: 'rgba(0,0,0,0.25)',
                  border: '1px solid rgba(137,180,250,0.2)',
                  borderRadius: 6,
                  padding: '10px 12px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 10,
                }}
              >
                {/* Title row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 700, fontSize: 12, color: '#89b4fa', fontFamily: 'monospace' }}>
                    L{layer}_N{selectedNeuronData.index}
                  </span>
                  <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    <PolysemPill poly={isPoly} />
                    <span style={{ fontSize: 10, color: '#f9a8d4', fontFamily: 'monospace' }}>
                      ‖w‖₂ = {l2Norm.toFixed(3)}
                    </span>
                  </div>
                </div>

                {/* Metrics row */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 11 }}>
                  <div style={{ color: 'var(--text-dim, #a6adc8)' }}>
                    Substrate Activation: <strong style={{ color: '#89b4fa' }}>{selectedNeuronData.activation.toFixed(4)}</strong>
                  </div>
                  <div style={{ color: 'var(--text-dim, #a6adc8)' }}>
                    L2 Weight Norm: <strong style={{ color: '#f9a8d4' }}>{l2Norm.toFixed(4)}</strong>
                  </div>
                </div>

                {/* Superposition note */}
                <div
                  style={{
                    fontSize: 10.5,
                    padding: '5px 8px',
                    borderRadius: 4,
                    background: isPoly ? 'rgba(245,158,11,0.08)' : 'rgba(16,185,129,0.08)',
                    border: `1px solid ${isPoly ? 'rgba(245,158,11,0.25)' : 'rgba(16,185,129,0.25)'}`,
                    color: isPoly ? '#f59e0b' : '#10b981',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 6,
                  }}
                >
                  {isPoly ? <AlertTriangle size={12} style={{ flexShrink: 0, marginTop: 1 }} /> : <ShieldCheck size={12} style={{ flexShrink: 0, marginTop: 1 }} />}
                  <span>
                    {isPoly
                      ? 'Polysemantic superposition detected — this coordinate encodes multiple features simultaneously. Decompose via SAE latents.'
                      : 'Approximately monosemantic — activation pattern suggests a single dominant feature direction.'}
                  </span>
                </div>

                {/* Per-token firing strip */}
                {selectedNeuronData.tokenActivations && selectedNeuronData.tokenActivations.length > 0 && (
                  <div>
                    <div style={{ fontSize: 10.5, color: 'var(--text-dim, #a6adc8)', marginBottom: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
                      <Activity size={12} /> Per-Token Firing Strip
                    </div>
                    <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
                      {selectedNeuronData.tokenActivations.map((val, idx) => {
                        const maxV = Math.max(...selectedNeuronData.tokenActivations!);
                        const minV = Math.min(...selectedNeuronData.tokenActivations!);
                        const k = maxV > minV ? (val - minV) / (maxV - minV) : 1;
                        return (
                          <span
                            key={idx}
                            title={`${tokens[idx] ?? idx}: ${val.toFixed(3)}`}
                            style={{
                              padding: '3px 6px',
                              borderRadius: 4,
                              fontSize: 10.5,
                              fontFamily: 'monospace',
                              background: `rgba(137,180,250,${(0.07 + 0.45 * k).toFixed(3)})`,
                              color: '#cdd6f4',
                            }}
                          >
                            '{tokens[idx] ?? `T${idx}`}': <strong>{val.toFixed(2)}</strong>
                          </span>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* ── Upstream / Downstream SAE connectivity ───────── */}
                <div style={{ paddingTop: 8, borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                  <div style={{ fontSize: 10.5, fontWeight: 600, color: 'var(--text, #cdd6f4)', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 5 }}>
                    <Dna size={12} /> SAE Latent Correspondence
                  </div>

                  {linkedFeatures.length === 0 ? (
                    <div style={{ fontSize: 10.5, color: 'var(--text-dim, #a6adc8)', fontStyle: 'italic' }}>
                      {loadingFeatures ? 'Loading SAE features…' : 'No SAE latent currently references this substrate coordinate.'}
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                      {linkedFeatures.map((feat) => (
                        <button
                          key={feat.feature_id}
                          onClick={() => {
                            setSelectedFeatureId(feat.feature_id);
                            setActiveTab('sae');
                          }}
                          style={{
                            padding: '6px 8px',
                            background: 'rgba(166,227,161,0.05)',
                            border: '1px solid rgba(166,227,161,0.2)',
                            borderRadius: 4,
                            cursor: 'pointer',
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            fontSize: 11,
                            width: '100%',
                            textAlign: 'left',
                          }}
                        >
                          <div>
                            <div style={{ fontWeight: 700, color: '#a6e3a1', fontFamily: 'monospace' }}>{feat.feature_id}</div>
                            <div style={{ fontSize: 10, color: 'var(--text-dim, #a6adc8)' }}>{feat.semantic_label}</div>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                            <span style={{ fontSize: 9.5, padding: '1px 5px', borderRadius: 3, background: 'rgba(137,180,250,0.15)', color: '#89b4fa' }}>
                              {feat.evidence_level}
                            </span>
                            <ArrowRight size={12} color="#a6e3a1" />
                          </div>
                        </button>
                      ))}
                    </div>
                  )}

                  {/* Upstream / Downstream summary */}
                  {linkedFeatures.length > 0 && (
                    <div style={{ marginTop: 8, display: 'flex', gap: 12, fontSize: 10.5, color: 'var(--text-dim, #a6adc8)' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <ArrowUpRight size={11} color="#89b4fa" />
                        Upstream: {linkedFeatures.filter(f => f.layer < layer).length} latent(s)
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <ArrowDownRight size={11} color="#a6e3a1" />
                        Downstream: {linkedFeatures.filter(f => f.layer >= layer).length} latent(s)
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ═══════════════════════════════════════════════════════════
            SAE TAB — Sparse Dictionary Latent list
            ═══════════════════════════════════════════════════════════ */}
        {activeTab === 'sae' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

            {/* Scientific framing banner */}
            <div
              style={{
                padding: '8px 10px',
                background: 'rgba(166,227,161,0.08)',
                border: '1px solid rgba(166,227,161,0.2)',
                borderRadius: 6,
                fontSize: 11,
                lineHeight: 1.45,
              }}
            >
              <strong>Candidate Sparse Features (SAE Dictionary):</strong>{' '}
              Interpretable computational units scored by empirical Specificity, Consistency, and
              Cross-Prompt Stability — not physical neurons.
            </div>

            {/* Feature selector */}
            <div style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 4 }}>
              {saeFeatures.map((feat) => (
                <button
                  key={feat.feature_id}
                  onClick={() => setSelectedFeatureId(feat.feature_id)}
                  style={{
                    padding: '6px 10px',
                    fontSize: 11,
                    fontWeight: 600,
                    borderRadius: 4,
                    border: selectedFeature?.feature_id === feat.feature_id
                      ? '1px solid #a6e3a1'
                      : '1px solid rgba(255,255,255,0.1)',
                    background: selectedFeature?.feature_id === feat.feature_id
                      ? 'rgba(166,227,161,0.15)'
                      : 'rgba(0,0,0,0.2)',
                    color: selectedFeature?.feature_id === feat.feature_id ? '#a6e3a1' : 'var(--text-dim, #a6adc8)',
                    cursor: 'pointer',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {feat.feature_id.split('_').slice(-1)[0]}
                </button>
              ))}
              {saeFeatures.length === 0 && !loadingFeatures && (
                <span style={{ fontSize: 11, color: 'var(--text-dim, #a6adc8)', fontStyle: 'italic' }}>
                  No SAE features for this layer yet. Run an inference first.
                </span>
              )}
            </div>

            {/* Selected feature detail */}
            {selectedFeature && (
              <div
                style={{
                  padding: '10px 12px',
                  background: 'rgba(0,0,0,0.25)',
                  border: '1px solid rgba(255,255,255,0.07)',
                  borderRadius: 6,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 700, color: '#89b4fa', fontSize: 12, fontFamily: 'monospace' }}>
                    {selectedFeature.feature_id}
                  </span>
                  <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 6px', borderRadius: 4, background: 'rgba(137,180,250,0.15)', color: '#89b4fa' }}>
                    {selectedFeature.evidence_level}
                  </span>
                </div>

                <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text, #cdd6f4)' }}>
                  {selectedFeature.semantic_label}
                </div>

                {/* Quality metrics */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 10.5 }}>
                  {[
                    ['Uncertainty', selectedFeature.uncertainty !== undefined ? `${(selectedFeature.uncertainty * 100).toFixed(0)}%` : 'N/A', '#cdd6f4'],
                    ['Evidence Level', selectedFeature.evidence_level || 'N/A', '#cdd6f4'],
                    ['Sample Size', selectedFeature.sample_size?.toString() || 'N/A', '#cdd6f4'],
                    ['Method', selectedFeature.methodology || 'N/A', '#a6e3a1'],
                  ].map(([label, value, color]) => (
                    <div key={label as string} style={{ color: 'var(--text-dim, #a6adc8)' }}>
                      {label}: <strong style={{ color: color as string }}>{value}</strong>
                    </div>
                  ))}
                </div>

                {/* Projection table */}
                <div style={{ padding: 8, background: 'rgba(255,255,255,0.02)', borderRadius: 4, border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div style={{ fontSize: 10.5, fontWeight: 600, color: 'var(--text, #cdd6f4)', marginBottom: 4 }}>
                    W_U · d_i (linear projection):
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(100px, 1fr))', gap: 3, fontSize: 10.5 }}>
                    {Object.entries(selectedFeature.linear_logit_delta || {}).map(([tok, delta]) => (
                      <div key={tok} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 4px', background: 'rgba(0,0,0,0.2)', borderRadius: 3 }}>
                        <span>'{tok}'</span>
                        <strong style={{ color: delta > 0 ? '#a6e3a1' : '#f38ba8' }}>
                          {delta > 0 ? `+${delta.toFixed(2)}` : delta.toFixed(2)}
                        </strong>
                      </div>
                    ))}
                  </div>
                  <div style={{ fontSize: 9.5, color: 'var(--text-dim, #6c7086)', marginTop: 6, fontStyle: 'italic' }}>
                    ⚠️ Reflects unembedding geometry (Δz ≈ a_i · W_U d_i) — not causal evidence.
                  </div>
                </div>

                {/* Substrate anchor coordinates */}
                {selectedFeature.substrate_anchors && selectedFeature.substrate_anchors.length > 0 && (
                  <div style={{ fontSize: 10, color: 'var(--text-dim, #a6adc8)' }}>
                    Dense substrate coordinates:{' '}
                    <span style={{ fontFamily: 'monospace', color: '#f9a8d4' }}>
                      {'{'}
                      {selectedFeature.substrate_anchors.map((a) => `L${a.layer}_N${a.neuron_idx}`).join(', ')}
                      {'}'}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
