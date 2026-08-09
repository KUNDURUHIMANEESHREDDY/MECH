import React from 'react';
import { useModel } from '../hooks/useModel';
import './BenchmarkSuiteView.css';

export const BenchmarkSuiteView: React.FC = () => {
  const { state: model } = useModel();
  return (
    <div className="benchmark-suite">
      <h1>Benchmark Suite</h1>
      <p>Model: {model.modelInfo?.model_name || 'Not loaded'}</p>
    </div>
  );
};
