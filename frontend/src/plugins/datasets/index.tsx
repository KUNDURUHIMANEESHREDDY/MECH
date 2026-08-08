import { pluginRegistry } from '../../panel-system/pluginRegistry';
import type { FC, PanelContext } from '../../shared/types';

const DatasetViewerBody: FC<PanelContext> = () => {
  return <div style={{ padding: 12, color: 'var(--text-muted)' }}>Dataset Viewer — connect dataset data source.</div>;
};

pluginRegistry.register({
  id: 'dataset_viewer',
  title: 'Dataset Viewer',
  icon: 'Database',
  category: 'datasets',
  resourceKinds: ['dataset'],
  defaultDock: 'center',
  Body: DatasetViewerBody,
});
