import React, { useState } from 'react';
import { Clapperboard } from 'lucide-react';

export default function LayerEvolutionPanel() {
  const [animating, setAnimating] = useState(false);

  return (
    <div className="panel layer-evolution-panel" data-testid="layer-evolution-panel">
      <div className="panel-header">
        <h3><Clapperboard size={13} style={{ verticalAlign: 'middle', marginRight: 6, color: 'var(--accent)' }} /> Layer Evolution Trajectory Animation</h3>
        <button className="btn" style={{ fontSize: '10px' }} onClick={() => setAnimating(!animating)}>
          {animating ? 'Pause' : 'Play Trajectory'}
        </button>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
          {animating ? 'Animating residual stream trajectory through Layers 0 -> 11...' : 'Trajectory paused.'}
        </p>
      </div>
    </div>
  );
}
