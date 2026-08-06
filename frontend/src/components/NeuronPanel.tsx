import React from 'react';
import { colors } from '../design/tokens/colors';
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
        <div style={{ fontSize: 11, fontWeight: 600, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>
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
        <div style={{ background: colors.surfacePearl, borderRadius: 8, padding: 12 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
            Metadata
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 12 }}>
            <span style={{ color: colors.inkMuted48 }}>Neuron</span>
            <span style={{ color: colors.bodyMuted }}>{sel.index}</span>
            <span style={{ color: colors.inkMuted48 }}>Activation</span>
            <span style={{ color: colors.bodyMuted }}>{sel.activation.toFixed(4)}</span>
            <span style={{ color: colors.inkMuted48 }}>Layer</span>
            <span style={{ color: colors.bodyMuted }}>—</span>
            <span style={{ color: colors.inkMuted48 }}>Head</span>
            <span style={{ color: colors.bodyMuted }}>—</span>
          </div>
        </div>
      )}

      {!sel && (
        <div style={{ color: colors.inkMuted48, fontSize: 12, padding: 8 }}>
          Click a neuron bar to see metadata.
        </div>
      )}
    </div>
  );
}
