import React, { useState } from 'react';
import { pluginRegistry } from '../../panel-system/pluginRegistry';
import type { FC, PanelContext } from '../../shared/types';
import { useModel } from '../../shared/hooks/useModel';
import { useWorkspaceStore } from '../../shared/stores/workspace';
import { useSelectionStore } from '../../shared/stores/selection';
import { AttentionHeatmap } from '../../components/visualizations/panels/AttentionHeatmap';

const AttentionHeatmapBody: FC<PanelContext> = () => {
  const { state: model } = useModel();
  const sel = useSelectionStore();
  const tokens = model.result?.tokens.map(t => t.text) ?? [];
  const layer = model.result?.layers[sel.layer ?? 0];
  const head = layer?.heads[sel.head ?? 0];
  const matrix = head?.attentionMatrix ?? [];
  const [hoveredToken, setHoveredToken] = useState<number | null>(null);
  return (
    <AttentionHeatmap
      matrix={matrix}
      tokens={tokens}
      hoveredToken={hoveredToken}
      onHoverToken={setHoveredToken}
    />
  );
};

pluginRegistry.register({
  id: 'attention_heatmap',
  title: 'Attention Heatmap',
  icon: 'Flame',
  category: 'attention',
  resourceKinds: ['model', 'head'],
  defaultDock: 'center',
  Body: AttentionHeatmapBody,
});
