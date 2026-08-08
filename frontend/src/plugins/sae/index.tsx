import { pluginRegistry } from '../../panel-system/pluginRegistry';
import type { FC, PanelContext } from '../../shared/types';

const SAEFeatureBody: FC<PanelContext> = () => {
  return <div style={{ padding: 12, color: 'var(--text-muted)' }}>SAE Feature Inspector — connect SAE data source.</div>;
};

pluginRegistry.register({
  id: 'sae_feature',
  title: 'SAE Feature Inspector',
  icon: 'Dna',
  category: 'sae',
  resourceKinds: ['sae', 'feature', 'model'],
  defaultDock: 'right',
  Body: SAEFeatureBody,
});
