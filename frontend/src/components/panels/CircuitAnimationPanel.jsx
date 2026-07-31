import React, { useState } from 'react';
import { Clapperboard, Play, Pause } from 'lucide-react';
import { visualizationEngine } from '../../services/visualizationEngine.js';

export default function CircuitAnimationPanel() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [step, setStep] = useState(8);
  const totalLayers = 12;

  const togglePlay = () => {
    if (isPlaying) {
      visualizationEngine.pausePlayback();
      setIsPlaying(false);
    } else {
      visualizationEngine.startPlayback();
      setIsPlaying(true);
    }
  };

  return (
    <div className="panel circuit-animation-panel" data-testid="circuit-animation-panel">
      <div className="panel-header">
        <h3><Clapperboard size={13} style={{ verticalAlign: 'middle', marginRight: 6, color: 'var(--accent)' }} /> Circuit Activation Propagation Scrubber</h3>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <div className="controls" style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '12px' }}>
          <button className="btn" onClick={togglePlay}>
            {isPlaying ? (
              <>
                <Pause size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} /> Pause
              </>
            ) : (
              <>
                <Play size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} /> Play
              </>
            )}
          </button>
          <span style={{ fontSize: '12px', color: '#94a3b8' }}>
            Layer Depth: {step} / {totalLayers}
          </span>
        </div>

        <input
          type="range"
          min="0"
          max={totalLayers}
          value={step}
          onChange={(e) => setStep(Number(e.target.value))}
          style={{ width: '100%', marginBottom: '12px' }}
        />

        <div className="frame-preview" style={{ padding: '12px', background: '#0f172a', borderRadius: '6px', border: '1px solid #1e293b' }}>
          <h4 style={{ margin: '0 0 6px 0', fontSize: '13px', color: '#38bdf8' }}>Cached Runtime Frame @ Layer {step}</h4>
          <p style={{ margin: 0, fontSize: '11px', color: '#94a3b8' }}>
            Activation Magnitude: <strong>{(step * 0.85).toFixed(2)}</strong> | Active Attn Heads: <strong>[H{step % 12}.{step % 4}]</strong>
          </p>
        </div>
      </div>
    </div>
  );
}
