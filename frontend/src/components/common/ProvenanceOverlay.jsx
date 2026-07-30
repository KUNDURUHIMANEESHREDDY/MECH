import React from 'react';

export default function ProvenanceOverlay({ model = 'GPT-2 Small', prompt = 'The capital of France is', checkpoint = 'sae_gpt2_l8.pt', paper = 'Wang et al. (2022)' }) {
  return (
    <div style={{ padding: '6px 10px', background: 'rgba(15, 23, 42, 0.9)', border: '1px solid #334155', borderRadius: '4px', fontSize: '10px', color: '#94a3b8', marginTop: '6px' }}>
      <span>Model: <strong style={{ color: '#f1f5f9' }}>{model}</strong></span> | 
      <span> Checkpoint: <strong style={{ color: '#38bdf8' }}>{checkpoint}</strong></span> | 
      <span> Paper: <strong style={{ color: '#a855f7' }}>{paper}</strong></span>
    </div>
  );
}
