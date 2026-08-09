import React from 'react';
import { useModel } from '../hooks/useModel';
import './TransformerExplorer.css';

export const TransformerExplorer: React.FC = () => {
  const { state: model } = useModel();
  return (
    <div className="transformer-explorer">
      <h1>Transformer Explorer</h1>
      <p>Model: {model.modelInfo?.model_name || 'Not loaded'}</p>
    </div>
  );
};
