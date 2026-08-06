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

  const handleDelete = async (id) => {
    if (api && api.deleteExperiment) {
      await api.deleteExperiment(id);
      const updated = await api.listExperiments();
      setExperiments(updated || []);
    }
  };

  return (
    <div>
      <div className="section-header">
        <h2>Experiments Matrix</h2>
        <p className="hint">Manage circuit ablation, activation patching, and SAE feature steering runs.</p>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <h3>New Experiment Run</h3>
        <div className="exp-form-row">
          <input type="text" className="input-text" placeholder="Experiment Title" value={title} onChange={e => setTitle(e.target.value)} />
          <input type="text" className="input-text" placeholder="Target Circuit" value={circuit} onChange={e => setCircuit(e.target.value)} />
          <button className="btn" onClick={handleCreate}>Create Run</button>
        </div>
      </div>

      <div className="card">
        <h3>Experiment Logs</h3>
        {experiments.length === 0 ? (
          <p className="hint">No experiments registered yet. Create one above!</p>
        ) : (
          <div className="experiment-grid">
            {experiments.map((exp) => (
              <div key={exp.id} className="card" style={{ marginBottom: 0 }}>
                <div className="exp-header">
                  <h4>{exp.title}</h4>
                  <span className={'status-badge ' + exp.status}>{exp.status}</span>
                </div>
                <div className="exp-details">
                  <div>Target Circuit: <strong>{exp.targetCircuit}</strong></div>
                  <div>Created: <strong>{new Date(exp.createdAt).toLocaleDateString()}</strong></div>
                </div>
                <div className="exp-details" style={{ marginTop: 8 }}>
                  <button className="btn btn-secondary btn-sm" onClick={() => handleDelete(exp.id)}>Delete</button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
