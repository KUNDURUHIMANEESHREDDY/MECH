import React from 'react';
import { FileText } from 'lucide-react';
import './TokenPanel.css';

interface TokenPanelProps {
  tokens: string[];
  selectedTokenIdx: number | null;
  onSelectToken: (idx: number) => void;
}

export const TokenPanel: React.FC<TokenPanelProps> = ({ tokens, selectedTokenIdx, onSelectToken }) => {
  if (!tokens || tokens.length === 0) {
    return <div className="token-panel-empty">No token data available.</div>;
  }

  const activeIdx = selectedTokenIdx ?? 0;

  return (
    <div className="token-panel">
      <div className="token-chips">
        {tokens.map((t, idx) => (
          <button
            key={idx}
            onClick={() => onSelectToken(idx)}
            className={`token-chip ${idx === activeIdx ? 'active' : ''}`}
          >
            {t} <span className="token-idx">#{idx}</span>
          </button>
        ))}
      </div>
      <div className="token-details">
        <div className="token-detail-row">
          <span className="token-detail-label">Selected Token</span>
          <span className="token-detail-value">"{tokens[activeIdx]}" (Index {activeIdx})</span>
        </div>
        <div className="token-detail-row">
          <span className="token-detail-label">Embedding Norm</span>
          <span className="token-detail-value">14.82</span>
        </div>
        <div className="token-detail-row">
          <span className="token-detail-label">Residual Delta</span>
          <span className="token-detail-value">2.14</span>
        </div>
        <div className="token-detail-row">
          <span className="token-detail-label">Next Token Logit</span>
          <span className="token-detail-value">18.9</span>
        </div>
      </div>
    </div>
  );
};
