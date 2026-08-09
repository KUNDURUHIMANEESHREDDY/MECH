import React from 'react';
import { useModel } from '../hooks/useModel';
import './LayerSidebar.css';

export const LayerSidebar: React.FC = () => {
  const { state: model } = useModel();
  return (
    <div className="layer-sidebar">
      <div className="layer-sidebar-header">
        <span>Layers</span>
      </div>
      <div className="layer-sidebar-empty">
        {model.loaded ? 'No layers loaded.' : 'Load a model to view layers.'}
      </div>
    </div>
  );
};
