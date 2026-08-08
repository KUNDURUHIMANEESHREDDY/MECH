import { pluginRegistry } from '../../panel-system/pluginRegistry';
import type { FC, PanelContext } from '../../shared/types';

const CircuitExplorerBody: FC<PanelContext> = () => {
  return <div style={{ padding: 12, color: 'var(--text-muted)' }}>Circuit explorer — connect backend circuit graph data to render.</div>;
};

pluginRegistry.register({
  id: 'circuit_explorer',
  title: 'Circuit Explorer',
  icon: 'Activity',
  category: 'circuits',
  resourceKinds: ['circuit', 'model'],
  defaultDock: 'center',
  Body: CircuitExplorerBody,
});
