import React, { useState } from 'react';
import { pluginRegistry } from '../../panel-system/pluginRegistry';
import type { FC, PanelContext } from '../../shared/types';
import { useModel } from '../../shared/hooks/useModel';
import { useSelectionStore } from '../../shared/stores/selection';
import { NeuronPanel } from '../../components/NeuronPanel';

/**
 * Neuron panel plugin body.
 *
 * Resolves the selected layer's neurons from the model result and passes them
 * to NeuronPanel — which presents physical neurons as Reference Substrate
 * Anchors (Lℓ_Ni), not standalone semantic concepts.
 *
 * Props passed through:
 *   layer      — required for "Lℓ_Ni" coordinate labelling and SAE feature fetch
 *   modelName  — required so NeuronPanel fetches the right layer's SAE features
 */
const NeuronPanelBody: FC<PanelContext> = () => {
  const { state: model } = useModel();
  const sel = useSelectionStore();

  const layerIdx = sel.layer ?? 0;
  const tokens = model.result?.tokens.map((t) => t.text) ?? [];
  const layer = model.result?.layers[layerIdx];
  const head = layer?.heads[sel.head ?? 0];
  const neurons = head?.neurons ?? [];
  const modelName = model.modelInfo?.model_name ?? 'gpt2';

  const [selectedNeuron, setSelectedNeuron] = useState<number | null>(sel.neuron);

  return (
    <NeuronPanel
      neurons={neurons}
      selectedNeuron={selectedNeuron}
      onSelectNeuron={(idx) => {
        setSelectedNeuron(idx);
        sel.setSelectedNeuron(layerIdx, idx);
      }}
      tokens={tokens}
      layer={layerIdx}
      modelName={modelName}
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
