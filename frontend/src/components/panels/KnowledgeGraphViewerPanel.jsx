import React, { useState } from 'react';
import { Brain } from 'lucide-react';

export default function KnowledgeGraphViewerPanel() {
  const [filter, setFilter] = useState('All');
  const items = [
    { id: 'fact_1', type: 'Neuron', label: 'Neuron L8_N402', value: 'Indirect Object Identifier' },
    { id: 'fact_2', type: 'Feature', label: 'SAE Feature #1402', value: 'Fires on name tokens in double-clause prompts' },
    { id: 'c_ioi', type: 'Circuit', label: 'IOI Circuit', value: 'Name Recognition Circuit' },
    { id: 'disc_1', type: 'Discovery', label: 'Discovery #1', value: 'Geographic capital retrieval path confirmed' }
  ];

  const filtered = filter === 'All' ? items : items.filter((i) => i.type === filter);

  return (
    <div className="panel knowledge-graph-panel" data-testid="knowledge-graph-panel">
      <div className="panel-header">
        <h3><Brain size={13} style={{ verticalAlign: 'middle', marginRight: 6, color: 'var(--accent)' }} /> Knowledge Graph Viewer</h3>
        <div style={{ display: 'flex', gap: '4px' }}>
          {['All', 'Neuron', 'Feature', 'Circuit', 'Discovery'].map((f) => (
            <button
              key={f}
              className={`btn ${filter === f ? 'active' : ''}`}
              style={{ fontSize: '10px', padding: '2px 4px' }}
              onClick={() => setFilter(f)}
            >
              {f}
            </button>
          ))}
        </div>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
          {filtered.map((item) => (
            <div key={item.id} style={{ padding: '8px', background: 'var(--bg-elev-2)', borderRadius: '6px', border: '1px solid var(--border)' }}>
              <span className="badge" style={{ fontSize: '10px', background: 'var(--purple)', color: 'var(--bg)', padding: '2px 4px', borderRadius: '3px' }}>
                {item.type}
              </span>
              <h5 style={{ margin: '4px 0', fontSize: '12px', color: 'var(--text)' }}>{item.label}</h5>
              <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)' }}>{item.value}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
