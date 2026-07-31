import React, { useState } from 'react';
import { FlaskConical } from 'lucide-react';
import { knowledgeExplorerEngine } from '../../services/knowledgeExplorerEngine.js';

export default function MechanismSimulatorPanel() {
  const [patchType, setPatchType] = useState('zero_ablation');
  const [result, setResult] = useState(null);

  const runSimulation = () => {
    const res = knowledgeExplorerEngine.simulateIntervention('n_402', patchType);
    setResult(res);
  };

  return (
    <div className="panel mechanism-simulator-panel" data-testid="mechanism-simulator-panel">
      <div className="panel-header">
        <h3><FlaskConical size={13} style={{ verticalAlign: 'middle', marginRight: 6, color: 'var(--accent)' }} /> Interactive Mechanism Simulator</h3>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
          <select value={patchType} onChange={(e) => setPatchType(e.target.value)} style={{ padding: '4px', borderRadius: '4px', background: '#1e293b', color: '#fff', border: '1px solid #334155' }}>
            <option value="zero_ablation">Zero Ablation</option>
            <option value="mean_ablation">Mean Ablation</option>
            <option value="rescale_patch">Rescale 2.0x</option>
          </select>
          <button className="btn btn-primary" onClick={runSimulation}>Simulate</button>
        </div>

        {result && (
          <div style={{ padding: '12px', background: '#020617', borderRadius: '6px', border: '1px solid #1e293b' }}>
            <h5 style={{ margin: '0 0 4px 0', color: '#10b981' }}>Simulation Output</h5>
            <p style={{ margin: 0, fontSize: '11px', color: '#94a3b8' }}>
              Simulated Logit Delta: <strong>{result.simulatedDelta}</strong> | Confidence: <strong>{(result.confidence * 100).toFixed(0)}%</strong>
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
