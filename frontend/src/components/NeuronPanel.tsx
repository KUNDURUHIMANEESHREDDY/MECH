import React from 'react';
import { Brain, Activity } from 'lucide-react';
import './NeuronPanel.css';

interface NeuronPanelProps {
  neurons: Array<{ index: number; activation: number; tokenActivations?: number[] }>;
  selectedNeuron: number | null;
  onSelectNeuron: (idx: number) => void;
  tokens?: string[];
}

export const NeuronPanel: React.FC<NeuronPanelProps> = ({
  neurons,
  selectedNeuron,
  onSelectNeuron,
  tokens = [],
}) => {
  if (!neurons || neurons.length === 0) {
    return (
      <div className="neuron-panel">
        <div className="neuron-panel-header">
          <Brain size={16} />
          <span>Neuron Inspector</span>
        </div>
        <div className="neuron-panel-body">
          <p className="neuron-empty">No neuron data available.</p>
        </div>
      </div>
    );
  }

  const selected = neurons[selectedNeuron ?? 0];

  return (
    <div className="neuron-panel">
      <div className="neuron-panel-header">
        <Brain size={16} />
        <span>Neuron Inspector</span>
      </div>

      <div className="neuron-panel-body">
        <div className="neuron-selector">
          <label className="neuron-label">Neuron</label>
          <select
            value={selectedNeuron ?? 0}
            onChange={(e) => onSelectNeuron(Number(e.target.value))}
            className="neuron-select"
          >
            {neurons.map((n) => (
              <option key={n.index} value={n.index}>
                Neuron {n.index}
              </option>
            ))}
          </select>
        </div>

        {selected && (
          <div className="neuron-details">
            <div className="neuron-stat">
              <span className="neuron-stat-label">Activation</span>
              <span className="neuron-stat-value">{selected.activation.toFixed(4)}</span>
            </div>
            <div className="neuron-stat">
              <Activity size={14} />
              <span className="neuron-stat-label">Top Tokens</span>
            </div>
            <div className="neuron-tokens">
              {selected.tokenActivations?.map((val, idx) => (
                <span key={idx} className="neuron-token-chip">
                  {tokens[idx] || `#${idx}`}: {val?.toFixed(2)}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
