import React, { useState } from 'react';

export default function LayerComparisonPanel() {
  const [layerA, setLayerA] = useState(4);
  const [layerB, setLayerB] = useState(8);

  return (
    <div className="panel-content layer-comparison-panel" data-testid="layer-comparison-panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h4>Layer Comparison View</h4>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <span>Layer A:</span>
          <select value={layerA} onChange={(e) => setLayerA(Number(e.target.value))}>
            {Array.from({ length: 12 }, (_, i) => <option key={i} value={i + 1}>Layer {i + 1}</option>)}
          </select>
          <span>VS Layer B:</span>
          <select value={layerB} onChange={(e) => setLayerB(Number(e.target.value))}>
            {Array.from({ length: 12 }, (_, i) => <option key={i} value={i + 1}>Layer {i + 1}</option>)}
          </select>
        </div>
      </div>

      <div className="side-by-side-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <div className="card" style={{ margin: 0 }}>
          <h5>Layer {layerA} Metrics</h5>
          <div className="spec-row"><span>MLP Norm:</span> <strong>{(14.2 + layerA).toFixed(1)}</strong></div>
          <div className="spec-row"><span>Attention Norm:</span> <strong>{(10.8 + layerA * 0.9).toFixed(1)}</strong></div>
          <div className="spec-row"><span>Top Neuron:</span> <strong>L{layerA}_N402</strong></div>
        </div>

        <div className="card" style={{ margin: 0 }}>
          <h5>Layer {layerB} Metrics</h5>
          <div className="spec-row"><span>MLP Norm:</span> <strong>{(14.2 + layerB).toFixed(1)}</strong></div>
          <div className="spec-row"><span>Attention Norm:</span> <strong>{(10.8 + layerB * 0.9).toFixed(1)}</strong></div>
          <div className="spec-row"><span>Top Neuron:</span> <strong>L{layerB}_N118</strong></div>
        </div>
      </div>
    </div>
  );
}
