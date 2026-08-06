import React from 'react';
import { Mic } from 'lucide-react';

export default function ConferenceModePanel() {
  return (
    <div className="panel conference-mode-panel" data-testid="conference-mode-panel">
      <div className="panel-header">
        <h3><Mic size={13} style={{ verticalAlign: 'middle', marginRight: 6, color: 'var(--accent)' }} /> Conference Mode Presentation Generator</h3>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <div style={{ padding: '16px', background: 'var(--bg-elev-2)', borderRadius: '8px', border: '1px solid var(--border)' }}>
          <h4 style={{ margin: '0 0 6px 0', color: 'var(--success)' }}>Live Keynote Mode Active</h4>
          <p style={{ margin: 0, fontSize: '12px', color: 'var(--text-muted)' }}>
            Streaming interactive circuit demonstrations directly to conference audience displays.
          </p>
        </div>
      </div>
    </div>
  );
}
