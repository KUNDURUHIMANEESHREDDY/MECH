import React from 'react';
import { useModel } from '../hooks/useModel';
import './SessionTimelineView.css';

export const SessionTimelineView: React.FC = () => {
  const { state: model } = useModel();
  return (
    <div className="session-timeline">
      <h1>Session Timeline</h1>
      <p>Model: {model.modelInfo?.model_name || 'Not loaded'}</p>
    </div>
  );
};
