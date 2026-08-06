import React, { useState } from 'react';
import { ArrowRight } from 'lucide-react';
import { selectionManager } from '../../utils/selectionManager';

export default function CircuitGraphPanel() {
  const [selectedNode, setSelectedNode] = useState('N_L8_N402');

  const GRAPH_NODES = [
    { id: 'T_France', type: 'Token', label: 'Token: "France"', layer: 0 },
    { id: 'N_L8_N402', type: 'Neuron', label: 'Neuron: L8_N402', layer: 8 },
    { id: 'F_1402', type: 'Feature', label: 'SAE Feat: #1402 (IOI)', layer: 8 },
    { id: 'H_L8_H9', type: 'Head', label: 'Attn Head: L8_H9', layer: 8 },
    { id: 'P_Paris', type: 'Prediction', label: 'Pred: " Paris"', layer: 12 }
  ];

  return (
    <div className="panel-content circuit-graph-panel" data-testid="circuit-graph-panel">
      <h4>Interactive Circuit Graph</h4>
      <p className="hint">Multi-node graph mapping: Token {'→'} Neuron {'→'} Feature {'→'} Attention Head {'→'} Output Prediction.</p>

      <div className="graph-container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px', background: 'var(--bg)', borderRadius: '6px' }}>
        {GRAPH_NODES.map((node, idx) => (
          <React.Fragment key={node.id}>
            <div
              className={`graph-node node-type-${node.type.toLowerCase()} ${selectedNode === node.id ? 'selected' : ''}`}
              onClick={() => {
                setSelectedNode(node.id);
                if (node.type === 'Neuron') selectionManager.setNeuron(8, 402);
                if (node.type === 'Feature') selectionManager.setFeature(1402, 'IOI');
              }}
            >
              <div className="node-type-tag">{node.type}</div>
              <div className="node-label">{node.label}</div>
            </div>
            {idx < GRAPH_NODES.length - 1 && (
              <div className="graph-connector-arrow" style={{ display: 'flex', alignItems: 'center' }}>
                <ArrowRight size={14} />
              </div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
