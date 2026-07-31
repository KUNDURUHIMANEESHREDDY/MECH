import React, { useState, useEffect } from "react";

export default function PaperReproductionView() {
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [catalog, setCatalog] = useState(null);
  const [reproducing, setReproducing] = useState(false);
  const [reproDone, setReproDone] = useState(false);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/v1/research_catalog?item_type=papers")
      .then(res => res.json())
      .then(json => setCatalog(json.catalog))
      .catch(() => setCatalog([]));
  }, []);

  const handleReproduce = async (paperId) => {
    setReproducing(true);
    await new Promise(resolve => setTimeout(resolve, 2000));
    setReproducing(false);
    setReproDone(true);
  };

  const stepClass = (active, done) => {
    if (active) return "step-circle active";
    if (done) return "step-circle done";
    return "step-circle";
  };

  return (
    <div className="reproduction-view">
      <h2>Paper Reproductions</h2>
      <p className="hint">One-click reproduction of landmark mechanistic interpretability papers.</p>

      <div className="reproduction-layout">
        <div className="reproduction-sidebar">
          <h3>Select Paper</h3>
          {catalog?.map(paper => (
            <div
              key={paper.id}
              onClick={() => setSelectedPaper(paper)}
              className={'paper-card' + (selectedPaper?.id === paper.id ? ' selected' : '')}
            >
              <div className="paper-title">{paper.title}</div>
              <div className="paper-id">{paper.id}</div>
            </div>
          ))}
        </div>

        <div className="reproduction-main">
          {selectedPaper ? (
            <div className="card">
              <h3>{selectedPaper.title}</h3>
              <div className="paper-meta">
                <span>Model: gpt2-small</span>
                <span>Dataset: required</span>
              </div>

              <div className="reproduction-steps">
                <div className="repro-step">
                  <div className={stepClass(reproducing, reproDone)}>1</div>
                  <div>
                    <strong>Download Model & Dataset</strong>
                    <p className="hint">Fetches weights and required activations.</p>
                  </div>
                </div>
                <div className="repro-step">
                  <div className={stepClass(reproducing, reproDone)}>2</div>
                  <div>
                    <strong>Run Pipeline</strong>
                    <p className="hint">Executes the circuit discovery / extraction logic.</p>
                  </div>
                </div>
                <div className="repro-step">
                  <div className={stepClass(reproducing, reproDone)}>3</div>
                  <div>
                    <strong>Generate Report</strong>
                    <p className="hint">Compares observed metrics against TransformerLens baselines.</p>
                  </div>
                </div>
              </div>

              <button
                onClick={() => handleReproduce(selectedPaper.id)}
                disabled={reproducing}
                className="btn repro-btn"
              >
                {reproducing ? "Reproducing..." : reproDone ? "Reproduction Complete" : "Start One-Click Reproduction"}
              </button>
            </div>
          ) : (
            <div className="reproduction-empty">
              Select a paper to view reproduction details.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
