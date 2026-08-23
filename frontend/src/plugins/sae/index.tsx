import React, { useEffect, useState } from 'react';
import { pluginRegistry } from '../../panel-system/pluginRegistry';
import type { FC, PanelContext } from '../../shared/types';
import { useModel } from '../../shared/hooks/useModel';
import { useSelectionStore } from '../../shared/stores/selection';
import { scienceApi } from '../../science/api/scienceApi';
import { FeatureEvidence } from '../../science/types/scientificTypes';
import { SAEFeatureExplorer } from '../../components/SAEFeatureExplorer';

/**
 * SAE Feature panel plugin body.
 *
 * Fetches FeatureEvidence[] from the science API for the selected layer and
 * delegates all rendering to SAEFeatureExplorer — which presents sparse
 * dictionary latents with empirical quality scores, dense substrate
 * coordinates, and the W_U·d_i ≠ Δz comparison card.
 */
const SAEFeatureBody: FC<PanelContext> = () => {
  const { state: model } = useModel();
  const sel = useSelectionStore();

  const layers = model.result?.layers ?? [];
  const layerIdx = sel.layer ?? (layers.length > 0 ? 0 : 0);
  const modelName = model.modelInfo?.model_name ?? 'gpt2';

  const [features, setFeatures] = useState<FeatureEvidence[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    scienceApi
      .fetchLayerFeatures(layerIdx, modelName)
      .then((feats) => {
        setFeatures(feats);
        if (feats.length > 0 && !selectedId) {
          setSelectedId(feats[0].feature_id);
        }
      })
      .catch((err) => console.warn('SAEFeatureBody: feature fetch failed:', err))
      .finally(() => setLoading(false));
  }, [layerIdx, modelName]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div style={{ padding: 14, height: '100%', overflowY: 'auto', boxSizing: 'border-box' }}>
      <SAEFeatureExplorer
        features={features}
        selectedFeatureId={selectedId}
        onSelectFeature={setSelectedId}
        loading={loading}
        layer={layerIdx}
      />
    </div>
  );
};

pluginRegistry.register({
  id: 'sae_feature',
  title: 'SAE Feature Explorer',
  icon: 'Dna',
  category: 'sae',
  resourceKinds: ['sae', 'feature', 'model'],
  defaultDock: 'right',
  Body: SAEFeatureBody,
});