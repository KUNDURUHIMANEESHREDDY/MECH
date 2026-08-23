import React, { useState } from 'react';
import {
  GitCommit,
  GitMerge,
  ArrowRight,
  Plus,
  Minus,
  CheckCircle2,
  Layers,
  FileText,
} from 'lucide-react';
import { colors } from '../../design/tokens/colors';

export const MechanismDiffView: React.FC = () => {
  const [selectedV1, setSelectedV1] = useState(1);
  const [selectedV2, setSelectedV2] = useState(2);

  const diffData = {
    v1_number: 1,
    v2_number: 2,
    v1_title: 'v1: Single Head Name-Mover Hypothesis',
    v2_title: 'v2: Dual Name-Mover & Feedforward Routing Circuit',
    added_components: ['L10H0', 'MLP_L8'],
    removed_components: [],
    added_edges: [
      { source: 'L10H0', target: 'residual', relationship: 'CAUSAL' },
      { source: 'MLP_L8', target: 'residual', relationship: 'CANDIDATE' },
    ],
    new_evidence_citations: ['EVID-92 (L10H0 Causal Replication)', 'EVID-93 (MLP_L8 Observational Selectivity)'],
    rationale:
      'Incorporated secondary Name Mover head L10H0 after confirming Δlogit = +1.12 with Cohen\'s d = 2.89 over controls.',
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
              Mechanism Circuit Version Diff
            </h2>
          </div>
          <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 4 }}>
            Compare structural, causal, and evidence lineage between circuit revisions.
          </div>
        </div>

        {/* Version Switchers */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
          <span style={{ fontWeight: 600, color: colors.bodyMuted }}>Comparing:</span>
          <span
            style={{
              padding: '3px 8px',
              borderRadius: 4,
              backgroundColor: colors.surfaceTile1,
              border: `1px solid ${colors.hairline}`,
              fontWeight: 700,
            }}
          >
            v{diffData.v1_number}
          </span>
          <ArrowRight size={14} color={colors.bodyMuted} />
          <span
            style={{
              padding: '3px 8px',
              borderRadius: 4,
              backgroundColor: colors.primarySoft,
              color: colors.primary,
              border: `1px solid ${colors.primary}`,
              fontWeight: 700,
            }}
          >
            v{diffData.v2_number} (Active)
          </span>
        </div>
      </div>

      {/* Rationale Card */}
      <div
        style={{
          padding: 12,
          borderRadius: 8,
          backgroundColor: colors.surfaceTile1,
          border: `1px solid ${colors.hairline}`,
          fontSize: 12,
        }}
      >
        <span style={{ fontWeight: 700, color: colors.primary }}>Evolution Rationale: </span>
        <span style={{ color: colors.bodyText }}>{diffData.rationale}</span>
      </div>

      {/* Diff Sections */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
        {/* Component Modifications */}
        <div
          style={{
            padding: 14,
            borderRadius: 8,
            backgroundColor: colors.surfaceTile1,
            border: `1px solid ${colors.hairline}`,
            display: 'flex',
            flexDirection: 'column',
            gap: 10,
          }}
        >
          <div style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>
            Component Lineage Delta
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {diffData.added_components.map((comp) => (
              <div
                key={comp}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '6px 10px',
                  borderRadius: 4,
                  backgroundColor: colors.successSoft,
                  color: colors.successText,
                  fontSize: 12,
                  fontWeight: 600,
                  fontFamily: 'monospace',
                }}
              >
                <Plus size={14} /> Added Component: {comp}
              </div>
            ))}
          </div>
        </div>

        {/* Causal Edges & Evidence Delta */}
        <div
          style={{
            padding: 14,
            borderRadius: 8,
            backgroundColor: colors.surfaceTile1,
            border: `1px solid ${colors.hairline}`,
            display: 'flex',
            flexDirection: 'column',
            gap: 10,
          }}
        >
          <div style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>
            Causal Edges & Evidence Delta
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {diffData.added_edges.map((edge, i) => (
              <div
                key={i}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '6px 10px',
                  borderRadius: 4,
                  backgroundColor: colors.surfaceTile2,
                  border: `1px solid ${colors.hairline}`,
                  fontSize: 11,
                }}
              >
                <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>
                  +{edge.source} ➔ {edge.target}
                </span>
                <span
                  style={{
                    fontSize: 9,
                    fontWeight: 700,
                    padding: '2px 6px',
                    borderRadius: 3,
                    backgroundColor: colors.primarySoft,
                    color: colors.primary,
                  }}
                >
                  {edge.relationship}
                </span>
              </div>
            ))}

            {diffData.new_evidence_citations.map((evi, i) => (
              <div
                key={i}
                style={{
                  fontSize: 11,
                  color: colors.bodyMuted,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  marginTop: 2,
                }}
              >
                <CheckCircle2 size={13} color={colors.successText} />
                <span>Cited: {evi}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
