import React from 'react';

export default function ImmersiveCollabPanel() {
  return (
    <div className="panel immersive-collab-panel" data-testid="immersive-collab-panel">
      <div className="panel-header">
        <h3>🌐 Immersive Synchronized Collaboration</h3>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <p style={{ fontSize: '12px', color: '#94a3b8' }}>
          Synchronized view: 2 researchers exploring the same activation manifold in real time.
        </p>
      </div>
    </div>
  );
}
