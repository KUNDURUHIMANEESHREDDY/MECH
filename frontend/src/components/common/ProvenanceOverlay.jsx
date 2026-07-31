import React from 'react';

export default function ProvenanceOverlay({ model = 'GPT-2 Small', prompt = 'The capital of France is', checkpoint = 'sae_gpt2_l8.pt', paper = 'Wang et al. (2022)' }) {
  return (
    <div style={{ padding: '6px 10px', background: 'var(--bg-elev)', border: '1px solid var(--border)', borderRadius: '4px', fontSize: '10px', color: 'var(--text-dim)', marginTop: '6px' }}>
      <span>Model: <strong style={{ color: 'var(--text)' }}>{model}</strong></span> |
      <span> Checkpoint: <strong style={{ color: 'var(--accent)' }}>{checkpoint}</strong></span> |
      <span> Paper: <strong style={{ color: 'var(--accent-hover)' }}>{paper}</strong></span>
    </div>
  );
}
