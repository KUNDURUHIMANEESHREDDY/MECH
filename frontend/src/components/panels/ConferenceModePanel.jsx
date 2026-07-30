import React from 'react';

export default function ConferenceModePanel() {
  return (
    <div className="panel conference-mode-panel" data-testid="conference-mode-panel">
      <div className="panel-header">
        <h3>🎤 Conference Mode Presentation Generator</h3>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <div style={{ padding: '16px', background: '#020617', borderRadius: '8px', border: '1px solid #1e293b' }}>
          <h4 style={{ margin: '0 0 6px 0', color: '#10b981' }}>Live Keynote Mode Active</h4>
          <p style={{ margin: 0, fontSize: '12px', color: '#94a3b8' }}>
            Streaming interactive circuit demonstrations directly to conference audience displays.
          </p>
        </div>
      </div>
    </div>
  );
}
