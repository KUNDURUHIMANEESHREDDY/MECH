import React from 'react';
import { Target } from 'lucide-react';
import './PredictionPanel.css';

interface PredictionPanelProps {
  tokens: string[];
}

export const PredictionPanel: React.FC<PredictionPanelProps> = ({ tokens }) => {
  if (!tokens || tokens.length === 0) {
    return <div className="prediction-empty">No prediction data.</div>;
  }

  return (
    <div className="prediction-panel">
      <div className="prediction-header">
        <Target size={16} />
        <span>Prediction Inspector</span>
      </div>
      <div className="prediction-body">
        <div className="prediction-row">
          <span className="prediction-label">Prompt</span>
          <span className="prediction-value">{tokens.slice(0, 8).join(' ')}</span>
        </div>
        <div className="prediction-row">
          <span className="prediction-label">Top Predicted</span>
          <span className="prediction-value">{tokens[tokens.length - 1] || '—'}</span>
        </div>
      </div>
    </div>
  );
};
