import React, { useState } from 'react';
import { Zap } from 'lucide-react';
import { colors } from '../../design/tokens/colors';
import ProvenanceOverlay from '../common/ProvenanceOverlay';
import { sharedWorkspaceState } from '../../utils/sharedWorkspaceState';

export default function CausalGraphEditorPanel() {
  const [nodes, setNodes] = useState([
    { id: 'node_1', label: 'Layer 8 Head 4', pruned: false },
    { id: 'node_2', label: 'SAE Feature #1402', pruned: false },
  ]);

  const { provenance, annotations } = sharedWorkspaceState.getState();

  const togglePrune = (id) => {
    setNodes((prev) =>
      prev.map((n) => (n.id === id ? { ...n, pruned: !n.pruned } : n))
    );
  };

  return (
    <div className="panel causal-graph-editor-panel" data-testid="causal-graph-editor-panel">
      <div className="panel-header">
        <h3><Zap size={13} style={{ verticalAlign: 'middle', marginRight: 6, color: 'var(--accent)' }} /> Interactive Causal Graph Editor</h3>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {nodes.map((n) => (
            <div key={n.id} style={{ padding: '10px 12px', background: colors.surfacePearl, borderRadius: '6px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', opacity: n.pruned ? 0.5 : 1 }}>
              <span style={{ fontSize: '12px', color: colors.ink, textDecoration: n.pruned ? 'line-through' : 'none' }}>{n.label}</span>
              <button className="btn" style={{ fontSize: '10px', padding: '2px 8px' }} onClick={() => togglePrune(n.id)}>
                {n.pruned ? 'Restore Node' : 'Prune Node'}
              </button>
            </div>
          ))}
        </div>

        <div style={{ marginTop: '12px', paddingTop: '8px', borderTop: `1px solid ${colors.hairline}` }}>
          <h5 style={{ margin: '0 0 4px 0', fontSize: '11px', color: colors.purple }}>Annotations ({annotations.length})</h5>
          <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
            {annotations.map((a) => (
              <span key={a.id} className="badge" style={{ fontSize: '9px', background: colors.surfacePearl, color: colors.ink, padding: '2px 6px', borderRadius: '4px' }}>
                {a.target}: {a.text}
              </span>
            ))}
          </div>
        </div>

        <ProvenanceOverlay model={provenance.model} checkpoint={provenance.checkpoint} paper={provenance.paper} />
      </div>
    </div>
  );
}
