import React, { useEffect, useState } from 'react';
import {
  Dna, AlertTriangle, ShieldCheck, FlaskConical,
  ArrowRight, Zap, BarChart2, Layers
} from 'lucide-react';
import { colors } from '../design/tokens/colors';
import { FeatureEvidence } from '../science/types/scientificTypes';

/* ── Types ──────────────────────────────────────────────────────────── */

export interface SAEFeatureExplorerProps {
  /** SAE features for the selected layer, from scienceApi.fetchLayerFeatures(). */
  features: FeatureEvidence[];
  /** Currently selected feature id (controlled externally, or managed internally). */
  selectedFeatureId?: string | null;
  onSelectFeature?: (id: string) => void;
  loading?: boolean;
  layer?: number;
}

/* ── Evidence badge ─────────────────────────────────────────────────── */

const EVIDENCE_COLORS: Record<string, { bg: string; fg: string }> = {
  CAUSALLY_VERIFIED: { bg: colors.successSoft, fg: colors.successText },
  SUPPORTED:         { bg: colors.infoSoft,    fg: colors.infoText },
  CANDIDATE:         { bg: colors.warningSoft,  fg: colors.warningText },
  OBSERVED:          { bg: colors.surfacePearl, fg: colors.bodyMuted },
};

function EvidenceBadge({ level }: { level: string }) {
  const c = EVIDENCE_COLORS[level] ?? EVIDENCE_COLORS.OBSERVED;
  return (
    <span
      style={{
        fontSize: 9.5,
        fontWeight: 700,
        padding: '2px 6px',
        borderRadius: 3,
        background: c.bg,
        color: c.fg,
        letterSpacing: 0.3,
      }}
    >
      {level.replace('_', ' ')}
    </span>
  );
}

/* ── Score bar ──────────────────────────────────────────────────────── */

function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10.5, color: colors.bodyMuted }}>
        <span>{label}</span>
        <strong style={{ color: colors.ink }}>{(value * 100).toFixed(0)}%</strong>
      </div>
      <div style={{ height: 5, background: colors.surfacePearl, borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${(value * 100).toFixed(1)}%`, background: color, borderRadius: 3 }} />
      </div>
    </div>
  );
}

/* ── The W_U·d_i ≠ Δz comparison card ──────────────────────────────── */

function RepresentationalGeometryCard({ feature }: { feature: FeatureEvidence }) {
  const projEntries = Object.entries(feature.linear_logit_delta ?? {});
  const causalDelta = feature.causal_effect;

  return (
    <div
      style={{
        border: `1px solid ${colors.warningBorder}`,
        borderRadius: 8,
        overflow: 'hidden',
      }}
    >
      {/* Card header */}
      <div
        style={{
          background: colors.warningSoft,
          padding: '8px 12px',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          borderBottom: `1px solid ${colors.warningBorder}`,
        }}
      >
        <AlertTriangle size={14} color={colors.warning} />
        <span style={{ fontWeight: 700, fontSize: 11.5, color: colors.warningText }}>
          Representational Geometry vs. Causal Intervention
        </span>
      </div>

      {/* Two-column comparison */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr auto 1fr',
          gap: 0,
          background: colors.canvas,
        }}
      >
        {/* LEFT — W_U d_i (linear projection) */}
        <div style={{ padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div
            style={{
              fontSize: 10.5,
              fontWeight: 700,
              color: colors.infoText,
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              marginBottom: 2,
            }}
          >
            <BarChart2 size={12} />
            Directional Projection W_U · d_i
          </div>
          <div style={{ fontSize: 10, color: colors.bodyMuted, marginBottom: 4, fontStyle: 'italic' }}>
            Linear logit affinity — unembedding geometry
          </div>
          {projEntries.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {projEntries.map(([tok, delta]) => (
                <div
                  key={tok}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '3px 6px',
                    background: colors.surfaceTile1,
                    border: `1px solid ${colors.hairline}`,
                    borderRadius: 4,
                    fontSize: 11,
                  }}
                >
                  <span style={{ fontFamily: 'monospace', color: colors.body }}>'{tok}'</span>
                  <strong style={{ color: delta > 0 ? colors.success : colors.danger, fontFamily: 'monospace' }}>
                    {delta > 0 ? `+${delta.toFixed(3)}` : delta.toFixed(3)}
                  </strong>
                </div>
              ))}
            </div>
          ) : (
            <span style={{ fontSize: 11, color: colors.bodyMuted, fontStyle: 'italic' }}>No projection data.</span>
          )}
        </div>

        {/* SEPARATOR — ≠ */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '12px 8px',
            background: colors.warningSoft,
            borderLeft: `1px solid ${colors.warningBorder}`,
            borderRight: `1px solid ${colors.warningBorder}`,
          }}
        >
          <span
            style={{
              fontSize: 20,
              fontWeight: 900,
              color: colors.warning,
              lineHeight: 1,
            }}
          >
            ≠
          </span>
          <span style={{ fontSize: 8.5, color: colors.warningText, marginTop: 4, textAlign: 'center', maxWidth: 32 }}>
            NOT<br />EQUAL
          </span>
        </div>

        {/* RIGHT — Measured Causal Intervention Δz */}
        <div style={{ padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div
            style={{
              fontSize: 10.5,
              fontWeight: 700,
              color: causalDelta != null ? colors.successText : colors.bodyMuted,
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              marginBottom: 2,
            }}
          >
            <FlaskConical size={12} />
            Measured Causal Intervention Δz
          </div>
          <div style={{ fontSize: 10, color: colors.bodyMuted, marginBottom: 4, fontStyle: 'italic' }}>
            Ablation / path-patching effect
          </div>

          {causalDelta != null ? (
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: 4,
              }}
            >
              <div
                style={{
                  padding: '8px 10px',
                  background: colors.successSoft,
                  border: `1px solid ${colors.successBorder}`,
                  borderRadius: 6,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  <ShieldCheck size={13} color={colors.success} />
                  <span style={{ fontSize: 11, color: colors.successText, fontWeight: 600 }}>Causal Δz</span>
                </div>
                <strong
                  style={{
                    fontFamily: 'monospace',
                    fontSize: 14,
                    color: causalDelta > 0 ? colors.success : colors.danger,
                  }}
                >
                  {causalDelta > 0 ? `+${causalDelta.toFixed(3)}` : causalDelta.toFixed(3)}
                </strong>
              </div>
              <div style={{ fontSize: 10, color: colors.bodyMuted }}>
                Establishes actual causal effect — not a projection hypothesis.
              </div>
            </div>
          ) : (
            <div
              style={{
                padding: '8px 10px',
                background: colors.surfacePearl,
                border: `1px solid ${colors.hairline}`,
                borderRadius: 6,
                display: 'flex',
                flexDirection: 'column',
                gap: 4,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <Zap size={13} color={colors.bodyMuted} />
                <span style={{ fontSize: 11, color: colors.bodyMuted, fontWeight: 600 }}>Not yet measured</span>
              </div>
              <div style={{ fontSize: 10, color: colors.bodyMuted, fontStyle: 'italic' }}>
                Run a causal intervention to measure the actual Δz for this latent.
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer — epistemic statement */}
      <div
        style={{
          padding: '8px 12px',
          background: colors.warningSoft,
          borderTop: `1px solid ${colors.warningBorder}`,
          fontSize: 10.5,
          color: colors.warningText,
          lineHeight: 1.5,
        }}
      >
        <strong>Epistemic boundary: </strong>
        Linear logit affinity (W_U · d_i) is a <em>hypothesis</em> about representational geometry.
        Measured causal intervention (Δz) establishes the <em>actual causal effect</em>.
        These are formally distinct quantities and do not always agree.
      </div>
    </div>
  );
}

/* ── Main explorer component ─────────────────────────────────────────── */

export const SAEFeatureExplorer: React.FC<SAEFeatureExplorerProps> = ({
  features,
  selectedFeatureId: controlledId,
  onSelectFeature,
  loading = false,
  layer = 0,
}) => {
  const [internalId, setInternalId] = useState<string | null>(null);

  /* Sync with controlled prop or default to first feature. */
  useEffect(() => {
    if (controlledId !== undefined) return; // fully controlled
    if (features.length > 0 && !internalId) setInternalId(features[0].feature_id);
  }, [features]); // eslint-disable-line react-hooks/exhaustive-deps

  const activeId = controlledId !== undefined ? controlledId : internalId;
  const activeFeature = features.find((f) => f.feature_id === activeId) ?? features[0];

  const handleSelect = (id: string) => {
    setInternalId(id);
    onSelectFeature?.(id);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body }}>

      {/* ── Section header ──────────────────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <Dna size={15} color={colors.purple} />
          <span style={{ fontWeight: 700, color: colors.ink, fontSize: 13 }}>SAE Feature Explorer</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Layers size={12} color={colors.bodyMuted} />
          <span style={{ fontSize: 11, color: colors.bodyMuted }}>Layer {layer}</span>
          {loading && (
            <span style={{ fontSize: 10, color: colors.bodyMuted, fontStyle: 'italic' }}>Loading…</span>
          )}
        </div>
      </div>

      {/* ── Intro banner ─────────────────────────────────────────────── */}
      <div
        style={{
          padding: '8px 12px',
          background: colors.purpleSoft,
          border: `1px solid ${colors.purpleBorder}`,
          borderRadius: 6,
          fontSize: 11,
          lineHeight: 1.5,
          color: colors.purpleText,
        }}
      >
        <strong>Sparse Dictionary Latents:</strong>{' '}
        Each feature is an inferred computational unit that decomposes dense MLP activations
        into approximately-monosemantic directions. Scores below are empirical — not model-defined.
      </div>

      {/* ── Feature pill strip ──────────────────────────────────────── */}
      {features.length === 0 && !loading ? (
        <div style={{ fontSize: 12, color: colors.bodyMuted, fontStyle: 'italic' }}>
          No SAE features available for Layer {layer}. Run an inference to populate.
        </div>
      ) : (
        <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
          {features.map((feat) => {
            const isActive = feat.feature_id === activeId;
            const ec = EVIDENCE_COLORS[feat.evidence_level] ?? EVIDENCE_COLORS.OBSERVED;
            return (
              <button
                key={feat.feature_id}
                onClick={() => handleSelect(feat.feature_id)}
                style={{
                  padding: '5px 10px',
                  fontSize: 11,
                  fontWeight: 600,
                  borderRadius: 5,
                  border: isActive
                    ? `1.5px solid ${colors.purple}`
                    : `1px solid ${colors.hairline}`,
                  background: isActive ? colors.purpleSoft : colors.canvas,
                  color: isActive ? colors.purpleText : colors.bodyMuted,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 5,
                  transition: 'border-color 0.1s',
                }}
              >
                <span style={{ fontFamily: 'monospace' }}>{feat.feature_id.split('_').slice(-1)[0]}</span>
                <span style={{ width: 7, height: 7, borderRadius: '50%', background: ec.fg, flexShrink: 0 }} />
              </button>
            );
          })}
        </div>
      )}

      {/* ── Selected feature detail ──────────────────────────────────── */}
      {activeFeature && (
        <>
          {/* ── Identity + label ────────────────────────────────────── */}
          <div
            style={{
              border: `1px solid ${colors.hairline}`,
              borderRadius: 8,
              background: colors.canvas,
              padding: '12px 14px',
              display: 'flex',
              flexDirection: 'column',
              gap: 8,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
              <div>
                <div style={{ fontFamily: 'monospace', fontSize: 11, color: colors.bodyMuted, marginBottom: 2 }}>
                  Latent ID
                </div>
                <div style={{ fontWeight: 700, fontSize: 14, color: colors.purple, fontFamily: 'monospace' }}>
                  {activeFeature.feature_id}
                </div>
              </div>
              <EvidenceBadge level={activeFeature.evidence_level} />
            </div>

            {/* Inferred semantic label */}
            <div
              style={{
                padding: '8px 10px',
                background: colors.surfaceTile3,
                borderRadius: 5,
                border: `1px solid ${colors.hairline}`,
              }}
            >
              <div style={{ fontSize: 10, color: colors.bodyMuted, marginBottom: 2, textTransform: 'uppercase', letterSpacing: 0.3 }}>
                Inferred Semantic Label
              </div>
              <div style={{ fontSize: 13, fontWeight: 600, color: colors.ink }}>
                {activeFeature.semantic_label}
              </div>
            </div>
          </div>

          {/* ── Empirical quality scores ─────────────────────────────── */}
          <div
            style={{
              border: `1px solid ${colors.hairline}`,
              borderRadius: 8,
              background: colors.canvas,
              padding: '12px 14px',
              display: 'flex',
              flexDirection: 'column',
              gap: 10,
            }}
          >
            <div style={{ fontSize: 11, fontWeight: 700, color: colors.ink, textTransform: 'uppercase', letterSpacing: 0.4 }}>
              Empirical Quality Scores
            </div>
            <ScoreBar label="Evidence Level" value={activeFeature.evidence_level === 'CAUSALLY_VERIFIED' ? 1 : activeFeature.evidence_level === 'SUPPORTED' ? 0.75 : activeFeature.evidence_level === 'OBSERVED' ? 0.5 : 0.25} color={colors.primary} />
            <ScoreBar label="Uncertainty" value={activeFeature.uncertainty ?? 0.5} color={colors.success} />
            <ScoreBar label="Sample Size" value={Math.min((activeFeature.sample_size ?? 0) / 100, 1)} color={colors.purple} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginTop: 2, paddingTop: 8, borderTop: `1px solid ${colors.hairline}` }}>
              <span style={{ color: colors.bodyMuted }}>Methodology</span>
              <strong style={{ color: colors.success, fontFamily: 'monospace', fontSize: 10 }}>
                {activeFeature.methodology || 'N/A'}
              </strong>
            </div>
          </div>

          {/* ── Dense substrate coordinates ──────────────────────────── */}
          {activeFeature.substrate_anchors && activeFeature.substrate_anchors.length > 0 && (
            <div
              style={{
                border: `1px solid ${colors.hairline}`,
                borderRadius: 8,
                background: colors.canvas,
                padding: '12px 14px',
                display: 'flex',
                flexDirection: 'column',
                gap: 8,
              }}
            >
              <div style={{ fontSize: 11, fontWeight: 700, color: colors.ink, textTransform: 'uppercase', letterSpacing: 0.4 }}>
                Dense Substrate Coordinates
              </div>
              <div style={{ fontSize: 10.5, color: colors.bodyMuted, lineHeight: 1.5 }}>
                The dense MLP activations this latent most strongly depends on:
              </div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {activeFeature.substrate_anchors.map((a) => (
                  <div
                    key={`${a.layer}_${a.neuron_idx}`}
                    style={{
                      padding: '4px 9px',
                      background: colors.pinkSoft,
                      border: `1px solid ${colors.pinkBorder}`,
                      borderRadius: 5,
                      fontFamily: 'monospace',
                      fontSize: 11,
                      fontWeight: 700,
                      color: colors.pinkText,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                    }}
                  >
                    L{a.layer}_N{a.neuron_idx}
                    {a.weight_norm > 0 && (
                      <span style={{ fontSize: 9.5, fontWeight: 400, color: colors.bodyMuted }}>
                        ‖w‖{a.weight_norm.toFixed(2)}
                      </span>
                    )}
                    {a.is_polysemantic && (
                      <span style={{ fontSize: 8.5, fontWeight: 700, color: '#f59e0b' }}>POLY</span>
                    )}
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 10, color: colors.bodyMuted, fontStyle: 'italic' }}>
                <ArrowRight size={10} style={{ display: 'inline', marginRight: 3 }} />
                These are <strong>computational reference coordinates</strong>, not hardware neurons.
                Each L{'{layer}'}_N{'{idx}'} notation identifies a specific residual-stream dimension in the model's parameter space.
              </div>
            </div>
          )}

          {/* ── W_U·d_i ≠ Δz comparison card ───────────────────────── */}
          <RepresentationalGeometryCard feature={activeFeature} />
        </>
      )}
    </div>
  );
};
