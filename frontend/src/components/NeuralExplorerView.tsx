import React from 'react';
import { useModel } from '../hooks/useModel';
import './NeuralExplorerView.css';

export const NeuralExplorerView: React.FC = () => {
  const { state: model } = useModel();
  return (
    <div className="neural-explorer">
      <h1>Neural Explorer</h1>
      <p>Model: {model.modelInfo?.model_name || 'Not loaded'}</p>
    </div>
  );
};
