import { pluginRegistry } from '../../panel-system/pluginRegistry';
import type { FC, PanelContext } from '../../shared/types';

const EmbeddingViewerBody: FC<PanelContext> = () => {
  return <div style={{ padding: 12, color: 'var(--text-muted)' }}>Embedding Viewer — connect embedding data source.</div>;
};

pluginRegistry.register({
  id: 'embedding_viewer',
  title: 'Embedding Viewer',
  icon: 'Globe',
  category: 'embeddings',
  resourceKinds: ['model', 'token'],
  defaultDock: 'center',
  Body: EmbeddingViewerBody,
});
