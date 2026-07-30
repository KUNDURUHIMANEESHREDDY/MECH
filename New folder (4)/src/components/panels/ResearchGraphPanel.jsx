import React, { useState, useEffect } from 'react';
import { sharedWorkspaceState } from '../../utils/sharedWorkspaceState.js';
import { visualizationEngine } from '../../services/visualizationEngine.js';

export default function ResearchGraphPanel() {
  const [layout, setLayout] = useState('Hierarchical');
  const [nodes, setNodes] = useState([
    { id: 'q_1', type: 'Question', label: 'Why does GPT-2 predict Paris for capital of France?' },
    { id: 'h_1', type: 'Hypothesis', label: 'Layer 8 MLP Neuron #402 mediates geographic capital retrieval' },
    { id: 'e_1', type: 'Experiment', label: 'Activation Patching L8_N402 over clean vs corrupted prompts' },
    { id: 'ev_1', type: 'Evidence', label: 'Logit delta -4.2 on " Paris" upon zeroing L8_N402' },
    { id: 'c_1', type: 'Circuit', label: 'IOI Geographic Retrieval Circuit' },
    { id: 'conc_1', type: 'Conclusion', label: 'L8_N402 is an essential component of geographic capital prediction' }
  ]);

  const handleLayoutChange = (mode) => {
    setLayout(mode);
    visualizationEngine.setLayoutMode(mode);
  };

  const handleSelectNode = (n) => {
    visualizationEngine.setSelection({ selectedCircuit: n.id, selectedDiscovery: n.label });
  };

  return (
    <div className="panel research-graph-panel" data-testid="research-graph-panel">
      <div className="panel-header">
        <h3>🌐 Interactive Research Graph</h3>
        <div className="layout-controls" style={{ display: 'flex', gap: '6px' }}>
          {['Hierarchical', 'Force', 'Radial', 'Timeline'].map((m) => (
            <button
              key={m}
              className={`btn ${layout === m ? 'active' : ''}`}
              style={{ fontSize: '11px', padding: '2px 6px' }}
              onClick={() => handleLayoutChange(m)}
            >
              {m}
            </button>
          ))}
        </div>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <p className="hint">Navigating project DAG workspace layout: <strong>{layout}</strong></p>
        <div className="nodes-flow" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {nodes.map((n) => (
            <div
              key={n.id}
              onClick={() => handleSelectNode(n)}
              style={{
                padding: '8px 12px',
                borderRadius: '6px',
                background: '#1e293b',
                border: '1px solid #334155',
                cursor: 'pointer'
              }}
            >
              <span className="badge" style={{ fontSize: '10px', background: '#3b82f6', color: '#fff', padding: '2px 6px', borderRadius: '4px', marginRight: '8px' }}>
                {n.type}
              </span>
              <span style={{ fontSize: '12px', color: '#e2e8f0' }}>{n.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
