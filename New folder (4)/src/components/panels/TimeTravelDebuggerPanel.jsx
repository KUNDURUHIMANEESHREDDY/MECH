import React, { useState } from 'react';

export default function TimeTravelDebuggerPanel() {
  const [layer, setLayer] = useState(8);

  return (
    <div className="panel time-travel-debugger-panel" data-testid="time-travel-debugger-panel">
      <div className="panel-header">
        <h3>⏪ Time-Travel Activation Debugger</h3>
        <div style={{ display: 'flex', gap: '4px' }}>
          <button className="btn" style={{ fontSize: '10px' }} onClick={() => setLayer((l) => Math.max(0, l - 1))}>Step Back</button>
          <span style={{ fontSize: '11px', alignSelf: 'center', color: '#38bdf8' }}>Layer {layer}</span>
          <button className="btn" style={{ fontSize: '10px' }} onClick={() => setLayer((l) => Math.min(11, l + 1))}>Step Forward</button>
        </div>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <div style={{ padding: '12px', background: '#020617', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <h4 style={{ margin: '0 0 4px 0', color: '#f8fafc', fontSize: '13px' }}>Residual Stream State [Layer {layer}]</h4>
          <p style={{ margin: 0, fontSize: '11px', color: '#94a3b8' }}>
            Norm: <strong>14.28</strong> | Logit Lens Output: <strong>" Paris" (p=0.88)</strong>
          </p>
        </div>
      </div>
    </div>
  );
}
