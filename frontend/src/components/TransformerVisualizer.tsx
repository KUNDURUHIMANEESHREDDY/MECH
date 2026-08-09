import React from 'react';
import { useModel } from '../hooks/useModel';
import './TransformerVisualizer.css';

export const TransformerVisualizer: React.FC = () => {
  const { state: model } = useModel();
  return (
    <div className="transformer-visualizer">
      <h1>Transformer Visualizer</h1>
      <p>Model: {model.modelInfo?.model_name || 'Not loaded'}</p>
    </div>
  );
};
