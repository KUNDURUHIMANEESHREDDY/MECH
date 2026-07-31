import React from 'react';
import { ChartBar } from 'lucide-react';

export default function InteractiveFiguresPanel() {
  return (
    <div className="panel interactive-figures-panel" data-testid="interactive-figures-panel">
      <div className="panel-header">
        <h3><ChartBar size={13} style={{ verticalAlign: 'middle', marginRight: 6, color: 'var(--accent)' }} /> Interactive Publication Figures Studio</h3>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <div style={{ padding: '12px', background: '#020617', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <h4 style={{ margin: '0 0 4px 0', color: '#10b981', fontSize: '13px' }}>Figure 1: Live Causal Heatmap</h4>
          <p style={{ margin: '0 0 8px 0', fontSize: '11px', color: '#94a3b8' }}>
            Linked to raw tensor checkpoint <code>sae_gpt2_l8.pt</code>.
          </p>
          <button className="btn btn-primary" style={{ fontSize: '10px' }}>Export SVG / Vector LaTeX</button>
        </div>
      </div>
    </div>
  );
}
