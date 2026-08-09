import React from 'react';
import { Layers } from 'lucide-react';
import './LayerPanel.css';

interface LayerPanelProps {
  layerIdx: number;
  numHeads: number;
}

export const LayerPanel: React.FC<LayerPanelProps> = ({ layerIdx, numHeads }) => {
  return (
    <div className="layer-panel">
      <div className="layer-panel-header">
        <Layers size={16} />
        <span>Layer {layerIdx}</span>
      </div>
      <div className="layer-panel-body">
        <div className="layer-stat">
          <span className="layer-stat-label">Heads</span>
          <span className="layer-stat-value">{numHeads}</span>
        </div>
        <div className="layer-stat">
          <span className="layer-stat-label">Hidden Dim</span>
          <span className="layer-stat-value">768</span>
        </div>
      </div>
    </div>
  );
};
