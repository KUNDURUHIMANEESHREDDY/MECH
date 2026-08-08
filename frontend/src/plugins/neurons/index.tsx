import React, { useState } from 'react';
import { pluginRegistry } from '../../panel-system/pluginRegistry';
import type { FC, PanelContext } from '../../shared/types';
import { useModel } from '../../shared/hooks/useModel';
import { useSelectionStore } from '../../shared/stores/selection';
import { NeuronPanel } from '../../components/NeuronPanel';

const NeuronPanelBody: FC<PanelContext> = () => {
  const { state: model } = useModel();
  const sel = useSelectionStore();
  const tokens = model.result?.tokens.map(t => t.text) ?? [];
  const layer = model.result?.layers[sel.layer ?? 0];
  const head = layer?.heads[sel.head ?? 0];
  const neurons = head?.neurons ?? [];
  const [selectedNeuron, setSelectedNeuron] = useState<number | null>(sel.neuron);
  return (
    <NeuronPanel
      neurons={neurons}
      selectedNeuron={selectedNeuron}
      onSelectNeuron={(idx) => {
        setSelectedNeuron(idx);
        const layerIdx = sel.layer ?? 0;
        sel.setSelectedNeuron(layerIdx, idx);
      }}
      tokens={tokens}
    />
  );
};

pluginRegistry.register({
  id: 'neuron_panel',
  title: 'Neuron Inspector',
  icon: 'Zap',
  category: 'neurons',
  resourceKinds: ['neuron', 'model'],
  defaultDock: 'right',
  Body: NeuronPanelBody,
});
