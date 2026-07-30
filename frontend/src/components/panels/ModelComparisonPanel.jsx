import React, { useState, useEffect } from 'react';
import { visualizationService } from '../../services/visualizationService';

export default function ModelComparisonPanel() {
  const [modelA, setModelA] = useState('GPT-2 Small');
  const [modelB, setModelB] = useState('Pythia 160M');
  const [comparison, setComparison] = useState(null);

  useEffect(() => {
    visualizationService.fetchModelComparison('The capital of France is', modelA, modelB)
      .then((res) => setComparison(res))
      .catch(() => undefined);
  }, [modelA, modelB]);

  return (
    <div className="panel-content model-comparison-panel" data-testid="model-comparison-panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h4>Model Comparison Interface</h4>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <select value={modelA} onChange={(e) => setModelA(e.target.value)}>
            <option value="GPT-2 Small">GPT-2 Small</option>
            <option value="GPT-2 Medium">GPT-2 Medium</option>
          </select>
          <span>VS</span>
          <select value={modelB} onChange={(e) => setModelB(e.target.value)}>
            <option value="Pythia 160M">Pythia 160M</option>
            <option value="Llama-3 8B">Llama-3 8B</option>
          </select>
        </div>
      </div>

      {comparison ? (
        <div className="card" style={{ margin: 0 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', textAlign: 'center' }}>
            <div className="metric-box">
              <span className="hint">Cosine Similarity</span>
              <h3>{comparison.activations?.cosine_similarity ?? 0}</h3>
            </div>
            <div className="metric-box">
              <span className="hint">Head Alignment</span>
              <h3>{comparison.attention?.head_alignment_score ?? 0}</h3>
            </div>
            <div className="metric-box">
              <span className="hint">Top Prediction Match</span>
              <h3 style={{ color: comparison.predictions?.top_token_match ? 'var(--success)' : 'var(--danger)' }}>
                {comparison.predictions?.top_token_match ? 'MATCH' : 'MISMATCH'}
              </h3>
            </div>
          </div>
        </div>
      ) : (
        <p className="hint">Comparing model activation DTOs...</p>
      )}
    </div>
  );
}
