import React from 'react';
import { useModel } from '../hooks/useModel';
import './ModelsView.css';

export const ModelsView: React.FC = () => {
  const { state: model } = useModel();
  return (
    <div className="models-view">
      <h1>Models Catalog</h1>
      <p>Model: {model.modelInfo?.model_name || 'Not loaded'}</p>
    </div>
  );
};
