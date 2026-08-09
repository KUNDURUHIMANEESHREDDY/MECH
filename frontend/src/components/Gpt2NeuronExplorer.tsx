import React from 'react';
import { useModel } from '../hooks/useModel';
import './Gpt2NeuronExplorer.css';

export const Gpt2NeuronExplorer: React.FC = () => {
  const { state: model } = useModel();
  return (
    <div className="gpt2-neuron-explorer">
      <h1>GPT-2 Neuron Explorer</h1>
      <p>Model: {model.modelInfo?.model_name || 'Not loaded'}</p>
    </div>
  );
};
