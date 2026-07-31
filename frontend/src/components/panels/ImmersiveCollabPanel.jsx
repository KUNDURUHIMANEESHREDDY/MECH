import React from 'react';
import { Globe } from 'lucide-react';

export default function ImmersiveCollabPanel() {
  return (
    <div className="panel immersive-collab-panel" data-testid="immersive-collab-panel">
      <div className="panel-header">
        <h3><Globe size={13} style={{ verticalAlign: 'middle', marginRight: 6, color: 'var(--accent)' }} /> Immersive Synchronized Collaboration</h3>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <p style={{ fontSize: '12px', color: '#94a3b8' }}>
          Synchronized view: 2 researchers exploring the same activation manifold in real time.
        </p>
      </div>
    </div>
  );
}
