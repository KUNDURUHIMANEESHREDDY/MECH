import React from 'react';
import { useAppStore } from '../store/useAppStore';
import { useModel } from '../hooks/useModel';
import './Gpt2View.css';

export const Gpt2View: React.FC = () => {
  const { activePage, setActivePage } = useAppStore();
  const { state: model, infer } = useModel();
  const [prompt, setPrompt] = React.useState('The capital of France is');
  const [output, setOutput] = React.useState('');

  const handleRun = async () => {
    if (!model.loaded) return;
    const result = await infer(prompt);
    setOutput(result.generated_text);
  };

  return (
    <div className="gpt2-view">
      <div className="gpt2-header">
        <h1>GPT-2 Small</h1>
        <span className="gpt2-badge">12L / 768D / 12 Heads</span>
      </div>

      <div className="gpt2-section">
        <label className="gpt2-label">Prompt</label>
        <textarea
          className="gpt2-input"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
        />
        <button className="gpt2-run" onClick={handleRun} disabled={!model.loaded}>
          Run Inference
        </button>
      </div>

      {output && (
        <div className="gpt2-section">
          <label className="gpt2-label">Output</label>
          <div className="gpt2-output">{output}</div>
        </div>
      )}
    </div>
  );
};
