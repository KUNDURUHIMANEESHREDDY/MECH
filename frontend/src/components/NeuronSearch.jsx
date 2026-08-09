import React from 'react';
import { useModel } from '../hooks/useModel';
import './NeuronSearch.css';

export const NeuronSearch: React.FC = () => {
  const { state: model } = useModel();
  return (
    <div className="neuron-search">
      <h1>Neuron Search</h1>
      <p>Model: {model.modelInfo?.model_name || 'Not loaded'}</p>
    </div>
  );
};
