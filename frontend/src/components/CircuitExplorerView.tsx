import React from 'react';
import { useModel } from '../hooks/useModel';
import './CircuitExplorerView.css';

export const CircuitExplorerView: React.FC = () => {
  const { state: model } = useModel();
  return (
    <div className="circuit-explorer">
      <h1>Circuit Explorer</h1>
      <p>Model: {model.modelInfo?.model_name || 'Not loaded'}</p>
    </div>
  );
};
