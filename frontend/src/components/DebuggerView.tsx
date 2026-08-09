import React from 'react';
import { useModel } from '../hooks/useModel';
import './DebuggerView.css';

export const DebuggerView: React.FC = () => {
  const { state: model } = useModel();
  return (
    <div className="debugger-view">
      <h1>Debugger</h1>
      <p>Model: {model.modelInfo?.model_name || 'Not loaded'}</p>
    </div>
  );
};
