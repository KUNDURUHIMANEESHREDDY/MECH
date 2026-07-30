import React, { useState, useEffect } from 'react';
import { visualizationService } from '../../services/visualizationService';
import { selectionManager } from '../../utils/selectionManager';

export default function SAEFeaturePanel() {
  const [featureId, setFeatureId] = useState(1402);
  const [featureData, setFeatureData] = useState(null);

  useEffect(() => {
    visualizationService.fetchSAEFeature(featureId).then((res) => setFeatureData(res)).catch(() => undefined);
  }, [featureId]);

  return (
    <div className="panel-content sae-feature-panel" data-testid="sae-feature-panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <h4>Sparse Autoencoder (SAE) Feature Inspector</h4>
        <div style={{ display: 'flex', gap: '6px' }}>
          <button className="btn btn-secondary btn-sm" onClick={() => setFeatureId(1402)}>Feature #1402</button>
          <button className="btn btn-secondary btn-sm" onClick={() => setFeatureId(789)}>Feature #789</button>
        </div>
      </div>

      {featureData ? (
        <div className="sae-sections">
          <div className="sae-header-card" style={{ background: 'var(--bg)', padding: '10px', borderRadius: '6px', marginBottom: '10px' }}>
            <strong>{featureData.label || `Feature #${featureId}`}</strong>
            <p className="hint" style={{ margin: '4px 0 0' }}>
              Max Act: +{featureData.statistics?.max_activation ?? 0} | Firing Freq: {featureData.statistics?.firing_freq ?? 0}
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div className="sae-subcard">
              <h5>Connected Neurons</h5>
              {(featureData.connected_neurons || []).map((n, idx) => (
                <div key={idx} className="neuron-row" onClick={() => selectionManager.setNeuron(n.layer, n.neuron)}>
                  <span>Layer {n.layer}, Neuron {n.neuron}</span>
                  <span className="val">w: {n.weight}</span>
                </div>
              ))}
            </div>

            <div className="sae-subcard">
              <h5>Dataset High-Act Examples</h5>
              {(featureData.dataset_examples || []).map((ex, idx) => (
                <div key={idx} className="example-row" style={{ fontSize: '11px', padding: '4px', background: 'var(--bg)', borderRadius: '4px', marginBottom: '4px' }}>
                  "{ex}"
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <p className="hint">Loading SAE feature metadata...</p>
      )}
    </div>
  );
}
