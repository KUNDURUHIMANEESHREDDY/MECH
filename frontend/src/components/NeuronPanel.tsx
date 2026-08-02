import React from 'react';
import { NeuronData } from '../types';
import { ActivationHeatmap } from './visualizations/panels/ActivationHeatmap';

interface Props {
  neurons: NeuronData[];
  selectedNeuron: number | null;
  onSelectNeuron: (i: number) => void;
  tokens?: string[];
}

export function NeuronPanel({ neurons, selectedNeuron, onSelectNeuron, tokens }: Props) {
  const sel = selectedNeuron !== null ? neurons[selectedNeuron] : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div>
        <div style={{ fontSize: 11, fontWeight: 600, color: '#888', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>
          Activation Histogram
        </div>
        <ActivationHeatmap
          activations={neurons.map(n => n.activation)}
          neuronIndex={selectedNeuron}
          onSelectNeuron={onSelectNeuron}
          tokens={tokens}
          neuronTokenActivations={neurons.map(n => n.tokenActivations ?? null)}
        />
      </div>

      {sel && (
        <div style={{ background: '#1a1a2a', borderRadius: 8, padding: 12 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: '#888', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
            Metadata
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 12 }}>
            <span style={{ color: '#888' }}>Neuron</span>
            <span style={{ color: '#ddd' }}>{sel.index}</span>
            <span style={{ color: '#888' }}>Activation</span>
            <span style={{ color: '#ddd' }}>{sel.activation.toFixed(4)}</span>
            <span style={{ color: '#888' }}>Layer</span>
            <span style={{ color: '#ddd' }}>—</span>
            <span style={{ color: '#888' }}>Head</span>
            <span style={{ color: '#ddd' }}>—</span>
          </div>
        </div>
      )}

      {!sel && (
        <div style={{ color: '#666', fontSize: 12, padding: 8 }}>
          Click a neuron bar to see metadata.
        </div>
      )}
    </div>
  );
}
