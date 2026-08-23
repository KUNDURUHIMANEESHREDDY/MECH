import React from 'react';
import {
  X,
  ShieldCheck,
  ArrowRight,
  GitCommit,
  Layers,
  Activity,
  CheckCircle2,
  AlertTriangle,
  FileText,
  Sliders,
} from 'lucide-react';
import { colors } from '../../design/tokens/colors';
import { MethodologicalLimitationsBadge } from './MethodologicalLimitationsBadge';

interface Props {
  claimTitle: string;
  targetComponent: string;
  onClose: () => void;
}

export const ClaimDerivationModal: React.FC<Props> = ({ claimTitle, targetComponent, onClose }) => {
  const steps = [
    { type: 'OBSERVATION', title: 'Attention Routing', detail: 'L9H9 attends 87.4% to indirect object token position', status: 'VERIFIED' },
    { type: 'EXPERIMENT', title: 'Clean vs Corrupted Pairs', detail: 'IOI Template dataset with flipped names (Mary/John)', status: 'VERIFIED' },
    { type: 'INTERVENTION', title: 'Zero Ablation on L9H9', detail: 'Target logit dropped by Δlogit = +1.85 (causal reduction)', status: 'VERIFIED' },
    { type: 'CONTROL', title: 'Negative Control (L0H0)', detail: 'L0H0 ablation produced Δlogit = -0.04 (null baseline)', status: 'VERIFIED' },
    { type: 'REPLICATION', title: 'Deterministic Re-run', detail: 'Reproduced identically with Δdifferential < 1e-4', status: 'VERIFIED' },
    { type: 'EVIDENCE', title: 'EVID-91 Causal Link', detail: 'Cohen\'s d = 3.42 establishing high causal effect size', status: 'VERIFIED' },
  ];

  const alternatives = [
    { name: 'Alternative A: L8H4 Backup Head', status: 'CANDIDATE', detail: 'Showed weak attention (14.2%), Δlogit = 0.31' },
    { name: 'Alternative B: MLP Layer 8 Associative Memory', status: 'UNRESOLVED', detail: 'Direct feedforward contribution not yet ablated' },
    { name: 'Alternative C: Residual Stream Pre-computation', status: 'RULED_OUT', detail: 'Ablating early layers did not eliminate IO retrieval' },
  ];

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.65)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        padding: 20,
      }}
    >
      <div
        style={{
          width: '100%',
          maxWidth: 820,
          maxHeight: '90vh',
          backgroundColor: colors.canvas,
          borderRadius: 10,
          border: `1px solid ${colors.hairline}`,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          boxShadow: '0 12px 36px rgba(0,0,0,0.4)',
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: '16px 20px',
            borderBottom: `1px solid ${colors.hairline}`,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <ShieldCheck size={20} color={colors.primary} />
            <div>
              <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: colors.ink }}>
                Why is this claim valid?
              </h3>
              <div style={{ fontSize: 12, color: colors.bodyMuted }}>
                Scientific Derivation Chain & Epistemic Grounding for <strong>{targetComponent}</strong>
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              cursor: 'pointer',
              color: colors.bodyMuted,
            }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Content Body */}
        <div style={{ padding: 20, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 18 }}>
          {/* Claim Statement */}
          <div
            style={{
              padding: 14,
              borderRadius: 8,
              backgroundColor: colors.surfaceTile1,
              border: `1px solid ${colors.hairline}`,
            }}
          >
            <div style={{ fontSize: 11, fontWeight: 700, color: colors.primary, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Empirical Claim
            </div>
            <div style={{ fontSize: 14, fontWeight: 600, color: colors.ink, marginTop: 4 }}>
              "{claimTitle}"
            </div>
          </div>

          {/* Derivation Chain */}
          <div>
            <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 10 }}>
              Causal Derivation Pipeline
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {steps.map((s, idx) => (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12,
                    padding: '8px 12px',
                    borderRadius: 6,
                    backgroundColor: colors.surfaceTile2,
                    border: `1px solid ${colors.hairline}`,
                  }}
                >
                  <span
                    style={{
                      fontSize: 9,
                      fontWeight: 800,
                      padding: '2px 6px',
                      borderRadius: 4,
                      backgroundColor: colors.primarySoft,
                      color: colors.primary,
                      minWidth: 85,
                      textAlign: 'center',
                    }}
                  >
                    {s.type}
                  </span>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: colors.ink }}>{s.title}</div>
                    <div style={{ fontSize: 11, color: colors.bodyMuted }}>{s.detail}</div>
                  </div>
                  <CheckCircle2 size={16} color={colors.successText} />
                </div>
              ))}
            </div>
          </div>

          {/* Competing Alternative Explanations */}
          <div>
            <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 8 }}>
              Competing Alternative Explanations
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 8 }}>
              {alternatives.map((alt, i) => (
                <div
                  key={i}
                  style={{
                    padding: 10,
                    borderRadius: 6,
                    backgroundColor: colors.surfaceTile1,
                    border: `1px solid ${colors.hairline}`,
                    fontSize: 11,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 600, color: colors.ink }}>{alt.name}</span>
                    <span
                      style={{
                        fontSize: 9,
                        fontWeight: 700,
                        padding: '1px 5px',
                        borderRadius: 3,
                        backgroundColor:
                          alt.status === 'RULED_OUT'
                            ? colors.successSoft
                            : alt.status === 'UNRESOLVED'
                            ? colors.primarySoft
                            : colors.surfaceTile2,
                        color:
                          alt.status === 'RULED_OUT'
                            ? colors.successText
                            : alt.status === 'UNRESOLVED'
                            ? colors.primary
                            : colors.ink,
                      }}
                    >
                      {alt.status}
                    </span>
                  </div>
                  <div style={{ color: colors.bodyMuted, marginTop: 4 }}>{alt.detail}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Methodological Limitations Banner */}
          <MethodologicalLimitationsBadge method="ACTIVATION_PATCHING" />
        </div>
      </div>
    </div>
  );
};
