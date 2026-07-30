import React, { useState } from 'react';

export default function LayerEvolutionPanel() {
  const [animating, setAnimating] = useState(false);

  return (
    <div className="panel layer-evolution-panel" data-testid="layer-evolution-panel">
      <div className="panel-header">
        <h3>🎬 Layer Evolution Trajectory Animation</h3>
        <button className="btn" style={{ fontSize: '10px' }} onClick={() => setAnimating(!animating)}>
          {animating ? 'Pause' : 'Play Trajectory'}
        </button>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <p style={{ fontSize: '12px', color: '#94a3b8' }}>
          {animating ? 'Animating residual stream trajectory through Layers 0 ➔ 11...' : 'Trajectory paused.'}
        </p>
      </div>
    </div>
  );
}
