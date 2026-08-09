import React from 'react';
import { useModel } from '../hooks/useModel';
import './BenchmarkDashboard.css';

export const BenchmarkDashboard: React.FC = () => {
  const { state: model } = useModel();
  return (
    <div className="benchmark-dashboard">
      <h1>Benchmark</h1>
      <p>Model: {model.modelInfo?.model_name || 'Not loaded'}</p>
    </div>
  );
};
