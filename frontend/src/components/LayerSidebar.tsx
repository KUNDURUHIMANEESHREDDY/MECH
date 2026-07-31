import React from 'react';
import { LayerData } from '../types';

interface Props {
  layers: LayerData[];
  selectedLayer: number;
  selectedHead: number;
  onSelectLayer: (i: number) => void;
  onSelectHead: (i: number) => void;
}

export function LayerSidebar({ layers, selectedLayer, selectedHead, onSelectLayer, onSelectHead }: Props) {
  return (
    <div className="layer-list">
      <div className="layer-list-header">Layers</div>
      {layers.map((l) => (
        <div key={l.index}>
          <div
            onClick={() => onSelectLayer(l.index)}
            className={'layer-btn' + (selectedLayer === l.index ? ' active' : '')}
          >
            Layer {l.index}
          </div>
          {selectedLayer === l.index && (
            <div className="head-list">
              <div className="head-list-header">HEADS</div>
              {l.heads.map((h) => (
                <div
                  key={h.index}
                  onClick={() => onSelectHead(h.index)}
                  className={'head-btn' + (selectedHead === h.index ? ' active' : '')}
                >
                  Head {h.index}
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
