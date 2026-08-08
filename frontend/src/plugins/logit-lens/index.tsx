import { pluginRegistry } from '../../panel-system/pluginRegistry';
import type { FC, PanelContext } from '../../shared/types';

const LogitLensBody: FC<PanelContext> = () => {
  return <div style={{ padding: 12, color: 'var(--text-muted)' }}>Logit Lens — connect projection data source.</div>;
};

pluginRegistry.register({
  id: 'logit_lens',
  title: 'Logit Lens',
  icon: 'Search',
  category: 'logit-lens',
  resourceKinds: ['model', 'token'],
  defaultDock: 'bottom',
  Body: LogitLensBody,
});
