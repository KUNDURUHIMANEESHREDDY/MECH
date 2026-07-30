import React from 'react';

export default function ReportsView() {
  return (
    <div className="reports-view" data-testid="reports-view">
      <div className="section-header">
        <h2>Research & Interpretability Reports</h2>
        <p>Exportable markdown summaries, circuit diagrams, and metric visualizations.</p>
      </div>

      <div className="reports-grid">
        <div className="card">
          <h3>IOI Circuit Decomposition</h3>
          <p className="hint">Identified 7 heads responsible for indirect object identification in GPT-2 Small.</p>
          <div className="report-metrics">
            <div><span>Faithfulness:</span> <strong>94.2%</strong></div>
            <div><span>Completeness:</span> <strong>88.7%</strong></div>
          </div>
          <button className="btn btn-secondary btn-sm" style={{ marginTop: '12px' }}>Export Report (PDF/MD)</button>
        </div>

        <div className="card">
          <h3>SAE Feature Steering Analysis</h3>
          <p className="hint">Feature #1402 steering intensity vs output logit shift curve.</p>
          <div className="report-metrics">
            <div><span>Features Traced:</span> <strong>128</strong></div>
            <div><span>Max Delta:</span> <strong>+4.12</strong></div>
          </div>
          <button className="btn btn-secondary btn-sm" style={{ marginTop: '12px' }}>Export Report (PDF/MD)</button>
        </div>
      </div>
    </div>
  );
}
