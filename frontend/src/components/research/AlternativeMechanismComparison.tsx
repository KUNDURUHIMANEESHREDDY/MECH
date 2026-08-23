import React from 'react';
import {
  GitMerge,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  Layers,
  ArrowRight,
  Activity,
  BarChart2,
} from 'lucide-react';
import { colors } from '../../design/tokens/colors';

interface AlternativeHypothesis {
  id: string;
  name: string;
  targetComponent: string;
  status: 'PRIMARY_EXPLANATION' | 'SECONDARY_BACKUP' | 'UNRESOLVED' | 'RULED_OUT';
  observationalScore: number; // 0-100%
  causalDeltaLogit: number;   // e.g. 1.85
  controlContrast: number;    // Cohen's d
  falsificationTested: boolean;
  notes: string;
}

export const AlternativeMechanismComparison: React.FC = () => {
  const alternatives: AlternativeHypothesis[] = [
    {
      id: 'h1_primary',
      name: 'Primary Hypothesis (H1): L9H9 Name Mover',
      targetComponent: 'L9H9',
      status: 'PRIMARY_EXPLANATION',
      observationalScore: 87.4,
      causalDeltaLogit: 1.85,
      controlContrast: 3.42,
      falsificationTested: true,
      notes: 'Directly attends to indirect-object token; ablation produces largest causal prediction drop.',
    },
    {
      id: 'h2_alt_a',
      name: 'Alternative A: L8H4 Backup Head',
      targetComponent: 'L8H4',
      status: 'SECONDARY_BACKUP',
      observationalScore: 14.2,
      causalDeltaLogit: 0.31,
      controlContrast: 0.85,
      falsificationTested: true,
      notes: 'Weak baseline activity; partially compensates when primary head L9H9 is suppressed.',
    },
    {
      id: 'h3_alt_b',
      name: 'Alternative B: MLP Layer 8 Direct Computation',
      targetComponent: 'MLP_L8',
      status: 'UNRESOLVED',
      observationalScore: 42.0,
      causalDeltaLogit: 0.0,
      controlContrast: 0.0,
      falsificationTested: false,
      notes: 'Feedforward associative memory layer; knockout experiment not yet executed.',
    },
    {
      id: 'h4_alt_c',
      name: 'Alternative C: Residual Stream Pre-computation',
      targetComponent: 'Residual_L0_L4',
      status: 'RULED_OUT',
      observationalScore: 21.5,
      causalDeltaLogit: 0.04,
      controlContrast: 0.12,
      falsificationTested: true,
      notes: 'Early layer ablation failed to eliminate target retrieval without subsequent Name Movers.',
    },
  ];

  const getStatusBadge = (status: AlternativeHypothesis['status']) => {
    switch (status) {
      case 'PRIMARY_EXPLANATION':
        return { label: 'Primary Mechanism', bg: colors.successSoft, color: colors.successText };
      case 'SECONDARY_BACKUP':
        return { label: 'Secondary Backup', bg: colors.primarySoft, color: colors.primary };
      case 'UNRESOLVED':
        return { label: 'Unresolved', bg: colors.surfaceTile2, color: colors.bodyMuted };
      case 'RULED_OUT':
        return { label: 'Ruled Out (Falsified)', bg: colors.dangerSoft || colors.surfaceTile1, color: colors.dangerText || '#ef4444' };
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        backgroundColor: colors.canvas,
        color: colors.ink,
        padding: 20,
        overflowY: 'auto',
        gap: 16,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: `1px solid ${colors.hairline}`,
          paddingBottom: 14,
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <GitMerge size={18} color={colors.primary} />
            <h2 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>
              Alternative Mechanism Comparison
            </h2>
          </div>
          <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 4 }}>
            Empirical side-by-side evaluation of competing explanatory hypotheses for the circuit.
          </div>
        </div>
      </div>

      {/* Comparison Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 14 }}>
        {alternatives.map((alt) => {
          const badge = getStatusBadge(alt.status);

          return (
            <div
              key={alt.id}
              style={{
                padding: 16,
                borderRadius: 8,
                backgroundColor: colors.surfaceTile1,
                border: `1px solid ${colors.hairline}`,
                display: 'flex',
                flexDirection: 'column',
                gap: 12,
              }}
            >
              {/* Card Header */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>{alt.name}</div>
                  <div style={{ fontSize: 11, fontFamily: 'monospace', color: colors.primary, marginTop: 2 }}>
                    Component: {alt.targetComponent}
                  </div>
                </div>
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 700,
                    padding: '2px 6px',
                    borderRadius: 4,
                    backgroundColor: badge.bg,
                    color: badge.color,
                  }}
                >
                  {badge.label}
                </span>
              </div>

              {/* Empirical Metrics */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {/* Attention Routing Bar */}
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
                    <span style={{ color: colors.bodyMuted }}>Observational Routing:</span>
                    <strong>{alt.observationalScore}%</strong>
                  </div>
                  <div
                    style={{
                      height: 6,
                      borderRadius: 3,
                      backgroundColor: colors.surfaceTile2,
                      overflow: 'hidden',
                    }}
                  >
                    <div
                      style={{
                        height: '100%',
                        width: `${alt.observationalScore}%`,
                        backgroundColor: colors.primary,
                      }}
                    />
                  </div>
                </div>

                {/* Causal Delta Logit */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 11 }}>
                  <div style={{ padding: 8, borderRadius: 4, backgroundColor: colors.surfaceTile2 }}>
                    <div style={{ color: colors.bodyMuted, fontSize: 10 }}>Causal Δlogit</div>
                    <strong style={{ fontSize: 13, color: colors.ink }}>
                      {alt.causalDeltaLogit > 0 ? `+${alt.causalDeltaLogit}` : '0.00'}
                    </strong>
                  </div>
                  <div style={{ padding: 8, borderRadius: 4, backgroundColor: colors.surfaceTile2 }}>
                    <div style={{ color: colors.bodyMuted, fontSize: 10 }}>Control Contrast (d)</div>
                    <strong style={{ fontSize: 13, color: colors.ink }}>
                      {alt.controlContrast > 0 ? `${alt.controlContrast}` : 'N/A'}
                    </strong>
                  </div>
                </div>
              </div>

              {/* Notes & Falsification */}
              <div style={{ fontSize: 11, color: colors.bodyText, lineHeight: 1.4 }}>{alt.notes}</div>

              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  fontSize: 10,
                  color: alt.falsificationTested ? colors.successText : colors.bodyMuted,
                  borderTop: `1px solid ${colors.hairline}`,
                  paddingTop: 8,
                }}
              >
                {alt.falsificationTested ? <CheckCircle2 size={13} /> : <AlertTriangle size={13} />}
                <span>{alt.falsificationTested ? 'Falsification criterion evaluated' : 'Falsification test pending'}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
