import React, { useState } from 'react';
import { Info, ChevronDown, ChevronUp, AlertCircle } from 'lucide-react';
import { colors } from '../../design/tokens/colors';

interface Props {
  method: 'ATTENTION' | 'ACTIVATION_PATCHING' | 'PROBE' | 'SAE' | 'ABLATION' | 'CIRCUIT';
}

const LIMITATIONS_MAP: Record<string, { title: string; summary: string; details: string[] }> = {
  ATTENTION: {
    title: 'Attention Weight Caveat',
    summary: 'Attention weights are observational and do not establish causal mediation.',
    details: [
      'High attention weight does not imply the head causally influences downstream outputs.',
      'Information can be written into the residual stream earlier and simply routed through later heads.',
      'Always validate attention findings with activation patching and contrastive negative controls.',
    ],
  },
  ACTIVATION_PATCHING: {
    title: 'Activation Patching Scope',
    summary: 'Causal effects depend on clean/corrupted distribution pairing and metric choice.',
    details: [
      'Out-of-distribution activation replacement can introduce artificial model confusion.',
      'Negative control heads must be tested to ensure the effect is specific to the target component.',
      'Linear logit differences must be interpreted alongside probability shifts.',
    ],
  },
  PROBE: {
    title: 'Linear Probe Limitation',
    summary: 'Linear decodability does not prove the model uses the representation computationally.',
    details: [
      'Probes can extract information that is present in activations but ignored by subsequent layers.',
      'High probe accuracy requires causal intervention (e.g. steering vectors) for verification.',
    ],
  },
  SAE: {
    title: 'Sparse Autoencoder Interpretation Caveat',
    summary: 'Feature labels are candidate hypotheses until confirmed by targeted ablation.',
    details: [
      'Monosemantic feature interpretations can suffer from polysemantic residue.',
      'Feature activations should be intervened upon to confirm functional causality.',
    ],
  },
  ABLATION: {
    title: 'Ablation Sensitivity',
    summary: 'Mean/zero ablation can induce off-manifold behavior unless calibrated.',
    details: [
      'Zero ablation can push activations outside the training distribution.',
      'Use contrastive resample patching when possible to preserve on-manifold dynamics.',
    ],
  },
  CIRCUIT: {
    title: 'Circuit Completeness Limitation',
    summary: 'Circuit diagrams represent minimal sufficient subgraphs, not exhaustive computation.',
    details: [
      'Unmodeled backup pathways and diffuse residual contributions may exist.',
      'Faithfulness metrics should be measured on independent validation datasets.',
    ],
  },
};

export const MethodologicalLimitationsBadge: React.FC<Props> = ({ method }) => {
  const [expanded, setExpanded] = useState(false);
  const info = LIMITATIONS_MAP[method] || LIMITATIONS_MAP.ACTIVATION_PATCHING;

  return (
    <div
      style={{
        borderRadius: 6,
        border: `1px solid ${colors.hairline}`,
        backgroundColor: colors.surfaceTile1,
        padding: '8px 12px',
        fontSize: 11,
        color: colors.ink,
      }}
    >
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          cursor: 'pointer',
          userSelect: 'none',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 600 }}>
          <AlertCircle size={13} color={colors.warningText || colors.primary} />
          <span>{info.title}:</span>
          <span style={{ fontWeight: 400, color: colors.bodyMuted }}>{info.summary}</span>
        </div>
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </div>

      {expanded && (
        <ul
          style={{
            margin: '8px 0 0 0',
            paddingLeft: 18,
            color: colors.bodyMuted,
            lineHeight: 1.5,
          }}
        >
          {info.details.map((d, i) => (
            <li key={i}>{d}</li>
          ))}
        </ul>
      )}
    </div>
  );
};
