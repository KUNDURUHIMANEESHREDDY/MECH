import React from 'react';
import {
  HelpCircle,
  FileCode2,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  Info,
  ArrowRight,
  ExternalLink,
  Target,
} from 'lucide-react';
import { colors } from '../../design/tokens/colors';

interface InspectorItem {
  id: string;
  title: string;
  type: string;
  status: string;
  whyDerived: string[];
  metrics: { label: string; value: string; source: 'COMPUTED' | 'LITERATURE_REPORTED' }[];
  limitations: string[];
  evidenceIds: string[];
}

interface Props {
  selectedItem?: InspectorItem;
  onOpenMethodology: () => void;
}

export const UniversalInspector: React.FC<Props> = ({
  selectedItem = {
    id: 'comp_L9H9',
    title: 'L9H9 (Name Mover Head)',
    type: 'ATTENTION_HEAD',
    status: 'CAUSAL_MATCH',
    whyDerived: [
      'Identified as top logit suppressor in unbiased layer 7–10 whole-model ablation scan.',
      'Δlogit = +2.13 on target indirect object token (Mary).',
      'Negative control L0H0 isolated (Δ = +0.11), rejecting non-specific degradation.',
      'Replicated across 3 independent evaluation seeds with deterministic torch states.',
    ],
    metrics: [
      { label: 'Treatment Δlogit', value: '+2.13', source: 'COMPUTED' },
      { label: 'Control Δlogit', value: '+0.11', source: 'COMPUTED' },
      { label: 'Effect Size (Cohen\'s d)', value: '3.42', source: 'COMPUTED' },
    ],
    limitations: [
      'Single-head knockout tests individual mediation only; does not isolate full 26-head circuit.',
      'Backup name-mover compensation partially buffers isolated head suppression.',
    ],
    evidenceIds: ['EVID-17', 'EVID-42', 'EVID-89'],
  },
  onOpenMethodology,
}) => {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        backgroundColor: colors.surfaceTile1,
        color: colors.ink,
        borderLeft: `1px solid ${colors.hairline}`,
        padding: 16,
        overflowY: 'auto',
        gap: 14,
        fontSize: 12,
      }}
    >
      {/* Inspector Title */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: `1px solid ${colors.hairline}`, paddingBottom: 10 }}>
        <div>
          <div style={{ fontSize: 11, color: colors.bodyMuted, textTransform: 'uppercase', fontWeight: 700 }}>
            Universal Scientific Inspector
          </div>
          <div style={{ fontSize: 13, fontWeight: 700, color: colors.ink, marginTop: 2 }}>
            {selectedItem.title}
          </div>
        </div>
        <span
          style={{
            fontSize: 10,
            fontWeight: 700,
            padding: '2px 6px',
            borderRadius: 4,
            backgroundColor: colors.successSoft,
            color: colors.successText,
          }}
        >
          {selectedItem.status}
        </span>
      </div>

      {/* "Why am I seeing this?" Derivation Chain */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <div style={{ fontWeight: 700, color: colors.ink, display: 'flex', alignItems: 'center', gap: 6 }}>
          <HelpCircle size={14} color={colors.primary} /> Why am I seeing this?
        </div>
        <div style={{ backgroundColor: colors.surfaceTile2, padding: 10, borderRadius: 6, display: 'flex', flexDirection: 'column', gap: 4 }}>
          {selectedItem.whyDerived.map((step, idx) => (
            <div key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: 6, fontSize: 11 }}>
              <span style={{ color: colors.primary, fontWeight: 700 }}>➔</span>
              <span style={{ color: colors.bodyText }}>{step}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Metrics & Numerical Provenance */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <div style={{ fontWeight: 700, color: colors.ink }}>
          Measured Metrics & Provenance
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 6 }}>
          {selectedItem.metrics.map((m, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                backgroundColor: colors.surfaceTile2,
                padding: '6px 10px',
                borderRadius: 4,
                fontSize: 11,
              }}
            >
              <span style={{ color: colors.bodyMuted }}>{m.label}:</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <strong>{m.value}</strong>
                <span
                  style={{
                    fontSize: 9,
                    fontWeight: 700,
                    padding: '1px 4px',
                    borderRadius: 3,
                    backgroundColor: colors.surfaceTile1,
                    color: colors.primary,
                  }}
                >
                  [{m.source}]
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Supporting Evidence Chain */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <div style={{ fontWeight: 700, color: colors.ink }}>
          Supporting Empirical Evidence
        </div>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {selectedItem.evidenceIds.map((id) => (
            <span
              key={id}
              style={{
                fontSize: 10,
                fontWeight: 700,
                fontFamily: 'monospace',
                padding: '3px 8px',
                borderRadius: 4,
                backgroundColor: colors.surfaceTile2,
                color: colors.primary,
                border: `1px solid ${colors.hairline}`,
              }}
            >
              {id}
            </span>
          ))}
        </div>
      </div>

      {/* Methodological Limitations */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <div style={{ fontWeight: 700, color: colors.ink, display: 'flex', alignItems: 'center', gap: 6 }}>
          <AlertTriangle size={14} color={colors.warningText || colors.primary} /> Scientific Limitations
        </div>
        <ul style={{ margin: 0, paddingLeft: 16, color: colors.bodyMuted, fontSize: 11 }}>
          {selectedItem.limitations.map((lim, idx) => (
            <li key={idx} style={{ marginBottom: 4 }}>{lim}</li>
          ))}
        </ul>
      </div>

      {/* 1-Click Methodology Button */}
      <button
        onClick={onOpenMethodology}
        style={{
          marginTop: 'auto',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 6,
          backgroundColor: colors.surfaceTile2,
          border: `1px solid ${colors.hairline}`,
          color: colors.ink,
          borderRadius: 6,
          padding: '8px 12px',
          fontSize: 11,
          fontWeight: 700,
          cursor: 'pointer',
        }}
      >
        <FileCode2 size={14} color={colors.primary} />
        View Exact Experimental Methodology
      </button>
    </div>
  );
};
