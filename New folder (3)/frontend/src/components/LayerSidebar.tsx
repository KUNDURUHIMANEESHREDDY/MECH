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
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: '#888', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
        Layers
      </div>
      {layers.map((l) => (
        <div key={l.index}>
          <div
            onClick={() => onSelectLayer(l.index)}
            style={{
              padding: '6px 10px',
              borderRadius: 6,
              background: selectedLayer === l.index ? '#3b82f6' : '#1e1e2e',
              color: selectedLayer === l.index ? '#fff' : '#ccc',
              cursor: 'pointer',
              fontSize: 13,
              fontWeight: selectedLayer === l.index ? 600 : 400,
              marginBottom: 4,
              transition: 'background 0.15s',
            }}
          >
            Layer {l.index}
          </div>
          {selectedLayer === l.index && (
            <div style={{ paddingLeft: 12, display: 'flex', flexDirection: 'column', gap: 2 }}>
              <div style={{ fontSize: 10, color: '#888', marginBottom: 2 }}>HEADS</div>
              {l.heads.map((h) => (
                <div
                  key={h.index}
                  onClick={() => onSelectHead(h.index)}
                  style={{
                    padding: '3px 8px',
                    borderRadius: 4,
                    background: selectedHead === h.index ? '#2d4a7a' : 'transparent',
                    color: selectedHead === h.index ? '#aac8ff' : '#999',
                    cursor: 'pointer',
                    fontSize: 11,
                    transition: 'background 0.15s',
                  }}
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
