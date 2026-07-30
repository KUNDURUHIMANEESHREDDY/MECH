import React, { useState } from 'react';

/**
 * KnowledgeGraphView - Scientific Knowledge Graph Exploration Environment
 *
 * Provides 4 exploratory perspectives across connected scientific entities:
 * 1. Mechanism View (Mechanism ➔ Evidence ➔ Experiments ➔ Replications)
 * 2. Paper View (Paper ➔ Claims ➔ Circuits ➔ Evidence)
 * 3. Feature View (Feature ➔ Neuron ➔ Circuit ➔ Claim)
 * 4. Campaign View (Campaign ➔ Experiments ➔ Evidence ➔ Publication)
 */

const PERSPECTIVES = {
  mechanism: {
    title: '🏛️ Mechanism View',
    subtitle: 'Trace mechanism claims to supporting evidence, experiment runs, and replication counts',
    flow: ['Mechanism Claim', 'Evidence Items', 'Experiment Runs', 'Replication Records'],
    nodes: [
      { id: 'c1', label: 'IOI Name Mover Circuit', type: 'Mechanism Claim', color: '#58a6ff', detail: 'Confidence: 0.962 | Status: Validated' },
      { id: 'e1', label: 'ACDC 94% Pruning Logit Recovery', type: 'Evidence', color: '#2ea043', detail: 'Fidelity: 0.972' },
      { id: 'e2', label: 'Causal Scrubbing Equivalence Pass', type: 'Evidence', color: '#2ea043', detail: '96% Accuracy Preserved' },
      { id: 'exp1', label: 'GPT2-S ACDC Edge Prune Run', type: 'Experiment', color: '#5cd4c4', detail: 'Runtime: 7.8s' },
      { id: 'r1', label: '14 Independent Cross-Model Replications', type: 'Replication', color: '#d0c0ff', detail: 'Models: GPT2, Gemma, Llama' }
    ]
  },
  paper: {
    title: '📄 Paper View',
    subtitle: 'Trace academic paper citations down to claims, minimal circuits, and evidence items',
    flow: ['Paper Citation', 'Derived Claims', 'Minimal Subgraphs', 'Supporting Evidence'],
    nodes: [
      { id: 'p1', label: 'Wang et al. (2022) IOI Paper', type: 'Paper', color: '#d0c0ff', detail: 'Citations: 142 | Landmark Mechanistic Paper' },
      { id: 'c1', label: 'IOI Circuit Model', type: 'Claim', color: '#58a6ff', detail: 'Name Mover & Backup Heads' },
      { id: 'sub1', label: '3-Head Minimal Subgraph', type: 'Circuit', color: '#feca57', detail: 'L9H9, L10H0, L11H10' },
      { id: 'ev1', label: 'Direct Logit Patching Proof', type: 'Evidence', color: '#2ea043', detail: 'Logit Diff: 3.55' }
    ]
  },
  feature: {
    title: '🧩 Feature & Neuron View',
    subtitle: 'Trace SAE sparse features down to individual attention heads, circuits, and high-level claims',
    flow: ['SAE Feature', 'Attention Head / Neuron', 'Circuit Subgraph', 'Mechanism Claim'],
    nodes: [
      { id: 'f1', label: 'SAE Feature #4096 (Proper Nouns)', type: 'SAE Feature', color: '#5cd4c4', detail: 'Sparsity L0: 0.0012' },
      { id: 'h1', label: 'GPT-2 L9H9 Attention Head', type: 'Attention Head', color: '#feca57', detail: 'Layer 9, Head 9' },
      { id: 'sub1', label: 'IOI Circuit Subgraph', type: 'Circuit', color: '#d0c0ff', detail: '3-Head Core' },
      { id: 'c1', label: 'IOI Name Mover Circuit', type: 'Claim', color: '#58a6ff', detail: 'Validated Claim' }
    ]
  },
  campaign: {
    title: '🧪 Campaign View',
    subtitle: 'Trace research campaigns to executed experiment matrices, evidence streams, and publications',
    flow: ['Research Campaign', 'Experiment Matrix', 'Evidence Stream', 'Published Manuscript'],
    nodes: [
      { id: 'camp1', label: 'IOI Cross-Model Campaign', type: 'Campaign', color: '#feca57', detail: 'Budget: $10.00 | Consumed: 2.07e13 FLOPs' },
      { id: 'exp1', label: '3 Executed Experiments', type: 'Experiment Matrix', color: '#5cd4c4', detail: 'Attribution, ACDC, Scrubbing' },
      { id: 'ev1', label: '3 Fused Evidence Items', type: 'Evidence Stream', color: '#2ea043', detail: 'Fused Confidence: 0.962' },
      { id: 'pub1', label: 'paper.tex Publication Package', type: 'Publication', color: '#d0c0ff', detail: 'Camera-Ready LaTeX Manuscript' }
    ]
  }
};

export default function KnowledgeGraphView({ api, onNavigate }) {
  const [activePerspective, setActivePerspective] = useState('mechanism');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedNode, setSelectedNode] = useState(null);

  const currentP = PERSPECTIVES[activePerspective] || PERSPECTIVES.mechanism;

  return (
    <div style={{ padding: 28, background: '#0b0b1a', color: '#e0e0ff', height: '100%', overflowY: 'auto', fontFamily: "'Inter', sans-serif" }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, margin: '0 0 6px 0', color: '#d0c0ff' }}>🕸️ Scientific Knowledge Graph</h1>
          <p style={{ margin: 0, color: '#888', fontSize: 13 }}>
            Unified provenance memory connecting Papers, Claims, Experiments, Circuits, Neurons, and Features
          </p>
        </div>
        <div style={{ background: '#1a1a2e', padding: '6px 14px', borderRadius: 8, border: '1px solid #3a3a5a', fontSize: 12 }}>
          <span style={{ color: '#888' }}>Graph Schema: </span>
          <span style={{ color: '#5cd4c4', fontWeight: 600 }}>13 Node Types, 11 Edge Types</span>
        </div>
      </div>

      {/* Perspective Switcher Tabs */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 20 }}>
        {[
          { key: 'mechanism', label: '🏛️ Mechanism View' },
          { key: 'paper', label: '📄 Paper View' },
          { key: 'feature', label: '🧩 Feature & Neuron View' },
          { key: 'campaign', label: '🧪 Campaign View' },
        ].map(p => (
          <button
            key={p.key}
            onClick={() => setActivePerspective(p.key)}
            style={{
              padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600,
              background: activePerspective === p.key ? '#2a2a5a' : '#151528',
              color: activePerspective === p.key ? '#5cd4c4' : '#888',
              transition: 'all 0.2s'
            }}
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* Perspective Info Header */}
      <div style={{ background: '#12122a', border: '1px solid #2a2a4a', borderRadius: 12, padding: 20, marginBottom: 20 }}>
        <h2 style={{ fontSize: 18, color: '#fff', margin: '0 0 4px 0' }}>{currentP.title}</h2>
        <div style={{ fontSize: 13, color: '#bbb', marginBottom: 16 }}>{currentP.subtitle}</div>

        {/* Provenance Flow Diagram */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', background: '#151528', padding: 12, borderRadius: 8, border: '1px solid #2a2a4a', flexWrap: 'wrap' }}>
          {currentP.flow.map((step, idx) => (
            <React.Fragment key={idx}>
              <span style={{ background: '#1a1a3a', border: '1px solid #5cd4c4', color: '#5cd4c4', padding: '4px 10px', borderRadius: 6, fontSize: 12, fontWeight: 700 }}>
                {step}
              </span>
              {idx < currentP.flow.length - 1 && <span style={{ color: '#888', fontWeight: 800 }}>➔</span>}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Graph Traversal Nodes Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 20 }}>
        
        {/* Left Column: Connected Graph Nodes */}
        <div style={{ background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', padding: 20 }}>
          <h3 style={{ fontSize: 14, color: '#d0c0ff', margin: '0 0 14px 0' }}>🔗 Connected Lineage Nodes</h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {currentP.nodes.map((node, idx) => (
              <div
                key={node.id}
                onClick={() => setSelectedNode(node)}
                style={{
                  background: '#151528',
                  border: `1px solid ${selectedNode?.id === node.id ? '#5cd4c4' : '#2a2a4a'}`,
                  borderRadius: 8,
                  padding: 14,
                  cursor: 'pointer',
                  borderLeft: `4px solid ${node.color}`,
                  transition: 'all 0.2s'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <span style={{ fontSize: 14, fontWeight: 700, color: '#fff' }}>{node.label}</span>
                  <span style={{ background: '#1a1a3a', color: node.color, padding: '2px 8px', borderRadius: 4, fontSize: 10, fontWeight: 700 }}>
                    {node.type}
                  </span>
                </div>
                <div style={{ fontSize: 12, color: '#888' }}>{node.detail}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Column: Node Inspector & Query Console */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          
          {/* Node Inspector */}
          <div style={{ background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', padding: 20 }}>
            <h3 style={{ fontSize: 14, color: '#5cd4c4', margin: '0 0 12px 0' }}>🔍 Node Lineage Inspector</h3>
            {selectedNode ? (
              <div style={{ background: '#151528', padding: 14, borderRadius: 8, border: '1px solid #2a2a4a' }}>
                <div style={{ fontSize: 16, fontWeight: 700, color: '#fff', marginBottom: 4 }}>{selectedNode.label}</div>
                <div style={{ fontSize: 12, color: selectedNode.color, fontWeight: 700, marginBottom: 8 }}>{selectedNode.type}</div>
                <div style={{ fontSize: 12, color: '#bbb' }}>{selectedNode.detail}</div>
              </div>
            ) : (
              <div style={{ fontSize: 12, color: '#888', fontStyle: 'italic' }}>
                Click any lineage node on the left to inspect its multi-hop connections and properties.
              </div>
            )}
          </div>

          {/* Graph Query Console */}
          <div style={{ background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', padding: 20 }}>
            <h3 style={{ fontSize: 14, color: '#58a6ff', margin: '0 0 10px 0' }}>🔎 First-Class Graph Queries</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {[
                'Show every experiment supporting IOI Name Mover',
                'Which papers discuss GPT-2 L9H9?',
                'Which SAE features appear in both GPT-2 and Gemma?'
              ].map((q, qIdx) => (
                <button
                  key={qIdx}
                  onClick={() => setSearchQuery(q)}
                  style={{
                    padding: '8px 12px', background: '#151528', border: '1px solid #2a2a4a', borderRadius: 6,
                    color: '#d0c0ff', cursor: 'pointer', fontSize: 12, textAlign: 'left', fontWeight: 500
                  }}
                >
                  ⚡ "{q}"
                </button>
              ))}
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}
