import React from 'react';

export default function ReportsView() {
  return (
    <div>
      <div className="section-header">
        <h2>Research & Interpretability Reports</h2>
        <p className="hint">Exportable markdown summaries, circuit diagrams, and metric visualizations.</p>
      </div>

      <div className="reports-grid">
        <div className="card">
          <h3>IOI Circuit Decomposition</h3>
          <p className="hint">Identified 7 heads responsible for indirect object identification in GPT-2 Small.</p>
          <div className="spec-row">
            <span>Confidence</span><span>96%</span>
          </div>
          <div className="spec-row">
            <span>Falsification</span><span>Passed (0 CE)</span>
          </div>
          <button className="btn btn-sm" style={{ marginTop: 8 }}>View Report</button>
        </div>

        <div className="card">
          <h3>Induction Heads Cross-Model</h3>
          <p className="hint">Replicated Induction Head Circuit across GPT-2, Gemma 2B, and Pythia 1B.</p>
          <div className="spec-row">
            <span>Universality</span><span>Consistent</span>
          </div>
          <button className="btn btn-sm" style={{ marginTop: 8 }}>View Report</button>
        </div>

        <div className="card">
          <h3>Greater-Than Circuit Ablation</h3>
          <p className="hint">Ablation study on Greater-Than temporal reasoning across GPT-2 variants.</p>
          <div className="spec-row">
            <span>Accuracy Drop</span><span>67% -&gt; 42%</span>
          </div>
          <button className="btn btn-sm" style={{ marginTop: 8 }}>View Report</button>
        </div>
      </div>
    </div>
  );
}
