import React, { useState } from 'react';

/**
 * EvidenceFusionView - Persistent Scientific Mechanism Claim Registry
 *
 * Displays accumulated scientific claims with long-term confidence scores,
 * replication counts, cross-model validations (GPT2, Gemma, Llama), supporting/contradicting
 * experiment counts, literature citations, and algorithm provenance.
 */

const MOCK_CLAIMS = [
  {
    id: 'claim_ioi_name_mover',
    title: 'IOI Name Mover Circuit',
    status: 'Validated',
    confidence: 0.962,
    replications: 14,
    models: ['GPT2-S', 'GPT2-M', 'Gemma2', 'Llama3'],
    supporting_experiments: 103,
    contradicting_experiments: 4,
    literature: [
      'Wang et al. 2022: Interpretability in the Wild',
      'Conmy et al. 2023: Automatic Circuit Discovery'
    ],
    algorithms: ['Attribution Patching', 'ACDC', 'Path Patching', 'Causal Scrubbing', 'Universality', 'Transcoders'],
    summary: 'L9H9 and L10H0 act as primary Name Mover Heads writing directly to IO token logits.',
    evidence: [
      { algorithm: 'Attribution Patching', score: 0.95, detail: 'Screened 144 heads; isolated L9H9 & L10H0 in O(1) pass.' },
      { algorithm: 'ACDC', score: 0.91, detail: 'Pruned 94% of edges; 3-head minimal subgraph recovers 97.2% logit diff.' },
      { algorithm: 'Path Patching', score: 0.94, detail: 'Direct intervention verified L9H9 -> L10H0 residual stream edge.' },
      { algorithm: 'Causal Scrubbing', score: 0.96, detail: 'Behavior preserved (96%) under name token-type equivalence class.' },
      { algorithm: 'Feature Universality', score: 0.88, detail: 'GPT-2 L9H9 aligned with Gemma L11H4 (0.88 cosine similarity).' },
      { algorithm: 'Transcoders', score: 0.92, detail: 'MLP L4 dictionary features explain 95% of target layer variance.' },
    ],
    graph: {
      nodes: [
        { id: 'T0', label: 'John gave a drink to Mary', type: 'Token' },
        { id: 'L9H9', label: 'Name Mover (L9H9)', type: 'Head' },
        { id: 'L10H0', label: 'Name Mover (L10H0)', type: 'Head' },
        { id: 'P0', label: 'Prediction: Mary', type: 'Output' },
      ],
      edges: [
        { source: 'T0', target: 'L9H9', weight: 0.94 },
        { source: 'L9H9', target: 'L10H0', weight: 0.91 },
        { source: 'L10H0', target: 'P0', weight: 0.96 },
      ]
    }
  },
  {
    id: 'claim_induction_heads',
    title: 'Induction Head Sequence Repeater',
    status: 'Validated',
    confidence: 0.941,
    replications: 22,
    models: ['GPT2-S', 'Gemma2', 'Llama3', 'Mistral7B'],
    supporting_experiments: 145,
    contradicting_experiments: 2,
    literature: [
      'Olsson et al. 2022: In-context Learning and Induction Heads'
    ],
    algorithms: ['Attribution Patching', 'ACDC', 'Causal Scrubbing', 'Universality'],
    summary: 'Previous-token head L4H2 attends to token K-1 while Induction Head L5H1 copies token K to current position.',
    evidence: [
      { algorithm: 'Attribution Patching', score: 0.93, detail: 'High attribution on L5H1 during prefix token repetition.' },
      { algorithm: 'ACDC', score: 0.89, detail: 'Minimal 2-head circuit recovers 94% of induction accuracy.' },
      { algorithm: 'Causal Scrubbing', score: 0.94, detail: 'Preserved behavior under repeated sequence equivalence class.' },
      { algorithm: 'Feature Universality', score: 0.91, detail: 'Gemma L5H1 matches GPT-2 L5H5 induction head (0.91 cosine).' },
    ],
    graph: {
      nodes: [
        { id: 'T0', label: '[A][B] ... [A]', type: 'Token' },
        { id: 'L4H2', label: 'Prev Token (L4H2)', type: 'Head' },
        { id: 'L5H1', label: 'Induction Head (L5H1)', type: 'Head' },
        { id: 'P0', label: 'Prediction: [B]', type: 'Output' },
      ],
      edges: [
        { source: 'T0', target: 'L4H2', weight: 0.91 },
        { source: 'L4H2', target: 'L5H1', weight: 0.95 },
        { source: 'L5H1', target: 'P0', weight: 0.93 },
      ]
    }
  }
];

export default function EvidenceFusionView({ api, onNavigate }) {
  const [selectedClaimId, setSelectedClaimId] = useState(MOCK_CLAIMS[0].id);

  const claim = MOCK_CLAIMS.find(c => c.id === selectedClaimId) || MOCK_CLAIMS[0];

  return (
    <div style={{ padding: 28, background: '#0b0b1a', color: '#e0e0ff', height: '100%', overflowY: 'auto', fontFamily: "'Inter', sans-serif" }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, margin: '0 0 6px 0', color: '#d0c0ff' }}>🏛️ Mechanism Claim Registry</h1>
          <p style={{ margin: 0, color: '#888', fontSize: 13 }}>
            Persistent scientific knowledge accumulation & cross-model evidence tracking
          </p>
        </div>
        <div style={{ background: '#1a1a2e', padding: '6px 14px', borderRadius: 8, border: '1px solid #3a3a5a', fontSize: 12 }}>
          <span style={{ color: '#888' }}>Registry Status: </span>
          <span style={{ color: '#2ea043', fontWeight: 600 }}>Active ({MOCK_CLAIMS.length} Claims)</span>
        </div>
      </div>

      {/* Claim Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {MOCK_CLAIMS.map(c => (
          <button
            key={c.id}
            onClick={() => setSelectedClaimId(c.id)}
            style={{
              padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600,
              background: selectedClaimId === c.id ? '#2a2a5a' : '#151528',
              color: selectedClaimId === c.id ? '#d0c0ff' : '#888',
              transition: 'all 0.2s'
            }}
          >
            {c.title} ({Math.round(c.confidence * 100)}%)
          </button>
        ))}
      </div>

      {/* Main Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 20 }}>
        
        {/* Left Column: Claim Knowledge Card */}
        <div style={{ background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', padding: 24 }}>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
            <div>
              <span style={{ fontSize: 11, color: '#5cd4c4', textTransform: 'uppercase', letterSpacing: 1.2, fontWeight: 700 }}>Scientific Mechanism Claim</span>
              <h2 style={{ fontSize: 20, color: '#fff', margin: '4px 0 8px 0' }}>{claim.title}</h2>
              <div style={{ fontSize: 13, color: '#bbb', lineHeight: 1.5 }}>{claim.summary}</div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: 28, fontWeight: 800, color: '#2ea043' }}>{claim.confidence.toFixed(3)}</div>
              <div style={{ fontSize: 11, color: '#888', textTransform: 'uppercase' }}>Composite Confidence</div>
            </div>
          </div>

          {/* Key Metrics Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 20, background: '#151528', padding: 14, borderRadius: 8, border: '1px solid #2a2a4a', textAlign: 'center' }}>
            <div>
              <div style={{ fontSize: 11, color: '#888', textTransform: 'uppercase' }}>Status</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#2ea043', marginTop: 2 }}>{claim.status}</div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: '#888', textTransform: 'uppercase' }}>Replications</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#feca57', marginTop: 2 }}>{claim.replications}</div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: '#888', textTransform: 'uppercase' }}>Supporting Exps</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#2ea043', marginTop: 2 }}>{claim.supporting_experiments}</div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: '#888', textTransform: 'uppercase' }}>Contradicting</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#f85149', marginTop: 2 }}>{claim.contradicting_experiments}</div>
            </div>
          </div>

          {/* Validated Models Badge Row */}
          <div style={{ marginBottom: 20 }}>
            <span style={{ fontSize: 11, color: '#888', textTransform: 'uppercase', display: 'block', marginBottom: 6 }}>Validated Models</span>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {claim.models.map(m => (
                <span key={m} style={{ background: '#1a1a3a', border: '1px solid #5cd4c4', color: '#5cd4c4', padding: '3px 10px', borderRadius: 6, fontSize: 12, fontWeight: 600 }}>
                  {m}
                </span>
              ))}
            </div>
          </div>

          {/* Used Algorithms Badge Row */}
          <div style={{ marginBottom: 24 }}>
            <span style={{ fontSize: 11, color: '#888', textTransform: 'uppercase', display: 'block', marginBottom: 6 }}>Proven Algorithm Suite</span>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {claim.algorithms.map(alg => (
                <span key={alg} style={{ background: '#1e1a2e', border: '1px solid #d0c0ff', color: '#d0c0ff', padding: '3px 10px', borderRadius: 6, fontSize: 11, fontWeight: 600 }}>
                  {alg}
                </span>
              ))}
            </div>
          </div>

          {/* Accumulated Evidence List */}
          <h3 style={{ fontSize: 14, color: '#d0c0ff', margin: '0 0 12px 0' }}>Accumulated Evidence</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {claim.evidence.map((ev, idx) => (
              <div key={idx} style={{ background: '#151528', border: '1px solid #2a2a4a', borderRadius: 8, padding: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ color: '#2ea043', fontWeight: 700 }}>✓</span>
                    <span style={{ fontSize: 13, fontWeight: 600, color: '#fff' }}>{ev.algorithm}</span>
                  </div>
                  <span style={{ fontSize: 12, fontWeight: 700, color: '#5cd4c4' }}>{ev.score.toFixed(2)}</span>
                </div>
                <div style={{ fontSize: 12, color: '#888', paddingLeft: 20 }}>{ev.detail}</div>
              </div>
            ))}
          </div>

        </div>

        {/* Right Column: Literature & Unified Computational Subgraph */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          
          {/* Literature Citations Card */}
          <div style={{ background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', padding: 20 }}>
            <h3 style={{ fontSize: 13, color: '#888', textTransform: 'uppercase', margin: '0 0 10px 0' }}>Associated Literature</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {claim.literature.map((lit, i) => (
                <div key={i} style={{ fontSize: 12, color: '#d0c0ff', background: '#151528', padding: '8px 12px', borderRadius: 6, border: '1px solid #2a2a4a' }}>
                  📄 {lit}
                </div>
              ))}
            </div>
          </div>

          {/* Unified Subgraph Visualizer */}
          <div style={{ background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', padding: 20, flex: 1 }}>
            <h3 style={{ fontSize: 13, color: '#888', textTransform: 'uppercase', margin: '0 0 12px 0' }}>Unified Computational Graph</h3>
            <div style={{ background: '#151528', borderRadius: 8, padding: 16, border: '1px solid #2a2a4a' }}>
              {claim.graph.nodes.map(n => (
                <div key={n.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', background: '#1a1a2e', borderRadius: 6, marginBottom: 8, borderLeft: `3px solid ${n.type === 'Head' ? '#5cd4c4' : n.type === 'Output' ? '#2ea043' : '#d0c0ff'}` }}>
                  <span style={{ fontSize: 13, color: '#fff', fontWeight: 600 }}>{n.label}</span>
                  <span style={{ fontSize: 10, color: '#888', textTransform: 'uppercase' }}>{n.type}</span>
                </div>
              ))}
            </div>

            {/* Quick Navigation Links */}
            {onNavigate && (
              <div style={{ marginTop: 16, display: 'flex', gap: 12 }}>
                <button
                  onClick={() => onNavigate('circuitexplorer')}
                  style={{ flex: 1, padding: 10, background: '#1a1a2e', border: '1px solid #3a3a5a', borderRadius: 6, color: '#5cd4c4', cursor: 'pointer', fontSize: 12, fontWeight: 600 }}
                >
                  🔍 Circuit Explorer →
                </button>
                <button
                  onClick={() => onNavigate('reasoning')}
                  style={{ flex: 1, padding: 10, background: '#1a1a2e', border: '1px solid #3a3a5a', borderRadius: 6, color: '#d0c0ff', cursor: 'pointer', fontSize: 12, fontWeight: 600 }}
                >
                  🔬 Reasoning Trace →
                </button>
              </div>
            )}
          </div>

        </div>

      </div>
    </div>
  );
}
