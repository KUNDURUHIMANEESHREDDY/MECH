import React, { useState, useEffect } from 'react';

export default function ExperimentsView({ api }) {
  const [experiments, setExperiments] = useState([]);
  const [title, setTitle] = useState('');
  const [circuit, setCircuit] = useState('Layer 9 Induction Heads');

  useEffect(() => {
    if (api && api.listExperiments) {
      api.listExperiments().then((res) => setExperiments(res || [])).catch(() => undefined);
    }
  }, [api]);

  const handleCreate = async () => {
    if (!title.trim()) return;
    const newExp = {
      id: `exp_${Date.now()}`,
      title,
      targetCircuit: circuit,
      status: 'active',
      createdAt: new Date().toISOString(),
      results: { meanLogitDiff: 2.14, patchedHeads: [8, 9] }
    };
    if (api && api.saveExperiment) {
      await api.saveExperiment(newExp);
      const updated = await api.listExperiments();
      setExperiments(updated || []);
    } else {
      setExperiments([newExp, ...experiments]);
    }
    setTitle('');
  };

  return (
    <div className="experiments-view" data-testid="experiments-view">
      <div className="section-header">
        <h2>Experiments Matrix</h2>
        <p>Manage circuit ablation, activation patching, and SAE feature steering runs.</p>
      </div>

      <div className="card" style={{ marginBottom: '20px' }}>
        <h3>New Experiment Run</h3>
        <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
          <input
            type="text"
            className="input-text"
            placeholder="Experiment Title (e.g. IOI Head 9.9 Ablation)"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            style={{ flex: 2 }}
          />
          <input
            type="text"
            className="input-text"
            placeholder="Target Circuit"
            value={circuit}
            onChange={(e) => setCircuit(e.target.value)}
            style={{ flex: 1 }}
          />
          <button className="btn btn-primary" onClick={handleCreate}>Create Run</button>
        </div>
      </div>

      <div className="card">
        <h3>Experiment Logs</h3>
        {experiments.length === 0 ? (
          <p className="hint">No experiments registered yet. Create one above!</p>
        ) : (
          <div className="experiment-grid">
            {experiments.map((exp) => (
              <div key={exp.id} className="experiment-card">
                <div className="exp-header">
                  <h4>{exp.title}</h4>
                  <span className={`status-badge ${exp.status}`}>{exp.status}</span>
                </div>
                <div className="exp-details">
                  <div>Target Circuit: <strong>{exp.targetCircuit}</strong></div>
                  <div>Created: <strong>{new Date(exp.createdAt).toLocaleDateString()}</strong></div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
