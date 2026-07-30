import React from 'react';

export default function ConfidenceHeatmapPanel() {
  return (
    <div className="panel confidence-heatmap-panel" data-testid="confidence-heatmap-panel">
      <div className="panel-header">
        <h3>🔥 Statistical Confidence Heatmap</h3>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '4px' }}>
          {[0.95, 0.98, 0.82, 0.91, 0.89, 0.96, 0.94, 0.97].map((c, i) => (
            <div key={i} style={{ padding: '12px', background: `rgba(16, 185, 129, ${c})`, color: '#0f172a', textAlign: 'center', borderRadius: '4px', fontWeight: 'bold', fontSize: '11px' }}>
              {(c * 100).toFixed(0)}%
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
