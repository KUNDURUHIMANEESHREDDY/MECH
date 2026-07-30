import React from 'react';

export default function DiscoveryComparisonPanel() {
  return (
    <div className="panel discovery-comparison-panel" data-testid="discovery-comparison-panel">
      <div className="panel-header">
        <h3>⚖️ Discovery Comparison Dashboard</h3>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
          <div style={{ padding: '10px', background: '#1e293b', borderRadius: '6px' }}>
            <h5 style={{ margin: '0 0 4px 0', color: '#38bdf8' }}>Discovery A (GPT-2)</h5>
            <p style={{ margin: 0, fontSize: '11px', color: '#94a3b8' }}>IOI Circuit p-val &lt; 0.001</p>
          </div>
          <div style={{ padding: '10px', background: '#1e293b', borderRadius: '6px' }}>
            <h5 style={{ margin: '0 0 4px 0', color: '#a855f7' }}>Discovery B (Gemma-2B)</h5>
            <p style={{ margin: 0, fontSize: '11px', color: '#94a3b8' }}>IOI Circuit r = 0.91 alignment</p>
          </div>
        </div>
      </div>
    </div>
  );
}
