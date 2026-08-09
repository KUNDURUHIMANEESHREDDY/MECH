import React from 'react';
import { useModel } from '../hooks/useModel';
import './KnowledgeGraphView.css';

export const KnowledgeGraphView: React.FC = () => {
  const { state: model } = useModel();
  return (
    <div className="knowledge-graph">
      <h1>Knowledge Graph</h1>
      <p>Model: {model.modelInfo?.model_name || 'Not loaded'}</p>
    </div>
  );
};
