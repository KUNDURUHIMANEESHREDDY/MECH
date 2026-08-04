import React, { useState } from 'react';
import { Glasses } from 'lucide-react';

export default function VR3DNetworkViewerPanel() {
  const [viewMode, setViewMode] = useState('3D');

  return (
    <div className="panel vr-3d-panel" data-testid="vr-3d-panel">
      <div className="panel-header">
        <h3><Glasses size={13} style={{ verticalAlign: 'middle', marginRight: 6, color: 'var(--accent)' }} /> VR / 3D Network Viewer</h3>
        <div style={{ display: 'flex', gap: '4px' }}>
          {['2D', '3D', 'VR Mode'].map((m) => (
            <button
              key={m}
              className={`btn ${viewMode === m ? 'active' : ''}`}
              style={{ fontSize: '10px', padding: '2px 6px' }}
              onClick={() => setViewMode(m)}
            >
              {m}
            </button>
          ))}
        </div>
      </div>
      <div className="panel-body" style={{ padding: '12px', textAlign: 'center' }}>
        <div style={{ padding: '24px', background: 'var(--bg-elev-2)', borderRadius: '8px', border: '1px dashed var(--border)' }}>
          <h4 style={{ margin: '0 0 8px 0', color: 'var(--accent)' }}>Interactive Network Render [{viewMode}]</h4>
          <p style={{ margin: 0, fontSize: '12px', color: 'var(--text-muted)' }}>
            {viewMode === 'VR Mode'
              ? 'VR Headset Connected. Walk inside GPT-2 Small activation manifold.'
              : 'Rotate, pan, and zoom 3D node-link network geometry.'}
          </p>
        </div>
      </div>
    </div>
  );
}
