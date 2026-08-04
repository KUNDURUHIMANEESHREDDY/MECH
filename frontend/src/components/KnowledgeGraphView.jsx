import React, { useState } from 'react';
import { ArrowRight } from 'lucide-react';
import { colors } from '../design/tokens/colors';

const PERSPECTIVES = {
  mechanism: {
    title: 'Mechanism View',
    subtitle: 'Trace mechanism claims to supporting evidence, experiment runs, and replication counts',
    flow: ['Mechanism Claim', 'Evidence Items', 'Experiment Runs', 'Replication Records'],
    nodes: [
      { id: 'c1', label: 'IOI Name Mover Circuit', type: 'Mechanism Claim', color: colors.primary, detail: 'Confidence: 0.962 | Status: Validated' },
      { id: 'e1', label: 'ACDC 94% Pruning Logit Recovery', type: 'Evidence', color: colors.success, detail: 'Fidelity: 0.972' },
      { id: 'e2', label: 'Causal Scrubbing Equivalence Pass', type: 'Evidence', color: colors.success, detail: '96% Accuracy Preserved' },
      { id: 'exp1', label: 'GPT2-S ACDC Edge Prune Run', type: 'Experiment', color: colors.primary, detail: 'Runtime: 7.8s' },
      { id: 'r1', label: '14 Independent Cross-Model Replications', type: 'Replication', color: colors.purpleBorder, detail: 'Models: GPT2, Gemma, Llama' }
    ]
  },
  paper: {
    title: 'Paper View',
    subtitle: 'Trace academic paper citations down to claims, minimal circuits, and evidence items',
    flow: ['Paper Citation', 'Derived Claims', 'Minimal Subgraphs', 'Supporting Evidence'],
    nodes: [
      { id: 'p1', label: 'Wang et al. (2022) IOI Paper', type: 'Paper', color: colors.purpleBorder, detail: 'Citations: 142 | Landmark Mechanistic Paper' },
      { id: 'c1', label: 'IOI Circuit Model', type: 'Claim', color: colors.primary, detail: 'Name Mover & Backup Heads' },
      { id: 'sub1', label: '3-Head Minimal Subgraph', type: 'Circuit', color: colors.warning, detail: 'L9H9, L10H0, L11H10' },
      { id: 'ev1', label: 'Direct Logit Patching Proof', type: 'Evidence', color: colors.success, detail: 'Logit Diff: 3.55' }
    ]
  },
  feature: {
    title: 'Feature & Neuron View',
    subtitle: 'Trace SAE sparse features down to individual attention heads, circuits, and high-level claims',
    flow: ['SAE Feature', 'Attention Head / Neuron', 'Circuit Subgraph', 'Mechanism Claim'],
    nodes: [
      { id: 'f1', label: 'SAE Feature #4096 (Proper Nouns)', type: 'SAE Feature', color: colors.primary, detail: 'Sparsity L0: 0.0012' },
      { id: 'h1', label: 'GPT-2 L9H9 Attention Head', type: 'Attention Head', color: colors.warning, detail: 'Layer 9, Head 9' },
      { id: 'sub1', label: 'IOI Circuit Subgraph', type: 'Circuit', color: colors.purpleBorder, detail: '3-Head Core' },
      { id: 'c1', label: 'IOI Name Mover Circuit', type: 'Claim', color: colors.primary, detail: 'Validated Claim' }
    ]
  },
  campaign: {
    title: 'Campaign View',
    subtitle: 'Trace research campaigns to executed experiment matrices, evidence streams, and publications',
    flow: ['Research Campaign', 'Experiment Matrix', 'Evidence Stream', 'Published Manuscript'],
    nodes: [
      { id: 'camp1', label: 'IOI Cross-Model Campaign', type: 'Campaign', color: colors.warning, detail: 'Budget: $10.00 | Consumed: 2.07e13 FLOPs' },
      { id: 'exp1', label: '3 Executed Experiments', type: 'Experiment Matrix', color: colors.primary, detail: 'Attribution, ACDC, Scrubbing' },
      { id: 'ev1', label: '3 Fused Evidence Items', type: 'Evidence Stream', color: colors.success, detail: 'Fused Confidence: 0.962' },
      { id: 'pub1', label: 'paper.tex Publication Package', type: 'Publication', color: colors.purpleBorder, detail: 'Camera-Ready LaTeX Manuscript' }
    ]
  }
};

export default function KnowledgeGraphView({ api, onNavigate }) {
  const [activePerspective, setActivePerspective] = useState('mechanism');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedNode, setSelectedNode] = useState(null);

  const currentP = PERSPECTIVES[activePerspective] || PERSPECTIVES.mechanism;

  return (
    <div className="explorer-page">
      <div className="kg-header">
        <div>
          <h1>Scientific Knowledge Graph</h1>
          <p className="hint">
            Unified provenance memory connecting Papers, Claims, Experiments, Circuits, Neurons, and Features
          </p>
        </div>
        <div className="kg-schema-badge">
          <span style={{ color: 'var(--text-dim)' }}>Graph Schema: </span>
          <span style={{ color: colors.primary, fontWeight: 600 }}>13 Node Types, 11 Edge Types</span>
        </div>
      </div>

      <div className="kg-perspective-tabs">
        {Object.entries(PERSPECTIVES).map(([key, p]) => (
          <button
            key={key}
            onClick={() => setActivePerspective(key)}
            className={'perspective-tab' + (activePerspective === key ? ' active' : '')}
          >
            {p.title}
          </button>
        ))}
      </div>

      <div className="kg-perspective-card">
        <h2>{currentP.title}</h2>
        <div className="hint">{currentP.subtitle}</div>

        <div className="flow-diagram">
          {currentP.flow.map((step, idx) => (
            <React.Fragment key={idx}>
              <span className="flow-step">{step}</span>
              {idx < currentP.flow.length - 1 && <span className="flow-arrow"><ArrowRight size={13} /></span>}
            </React.Fragment>
          ))}
        </div>
      </div>

      <div className="kg-grid">
        <div className="kg-column">
          <h3>Connected Lineage Nodes</h3>
          <div className="kg-nodes-list">
            {currentP.nodes.map((node) => (
              <div
                key={node.id}
                onClick={() => setSelectedNode(node)}
                className={'kg-node' + (selectedNode?.id === node.id ? ' selected' : '')}
                style={{ borderLeftColor: node.color }}
              >
                <div className="kg-node-header">
                  <span className="kg-node-label">{node.label}</span>
                  <span className="kg-node-type" style={{ color: node.color }}>{node.type}</span>
                </div>
                <div className="kg-node-detail">{node.detail}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="kg-side-column">
          <div className="kg-inspector-card">
            <h3>Node Lineage Inspector</h3>
            {selectedNode ? (
              <div className="kg-inspector-node">
                <div className="kg-inspector-label">{selectedNode.label}</div>
                <div className="kg-inspector-type" style={{ color: selectedNode.color }}>{selectedNode.type}</div>
                <div className="kg-inspector-detail">{selectedNode.detail}</div>
              </div>
            ) : (
              <div className="hint" style={{ fontStyle: 'italic' }}>
                Click any lineage node on the left to inspect its multi-hop connections and properties.
              </div>
            )}
          </div>

          <div className="kg-inspector-card">
            <h3>First-Class Graph Queries</h3>
            <div className="kg-queries">
              {[
                'Show every experiment supporting IOI Name Mover',
                'Which papers discuss GPT-2 L9H9?',
                'Which SAE features appear in both GPT-2 and Gemma?'
              ].map((q, i) => (
                <button key={i} onClick={() => setSearchQuery(q)} className="kg-query-btn">
                  "{q}"
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
