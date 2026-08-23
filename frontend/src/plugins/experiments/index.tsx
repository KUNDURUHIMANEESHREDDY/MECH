import React from 'react';
import { pluginRegistry } from '../../panel-system/pluginRegistry';
import type { FC, PanelContext } from '../../shared/types';
import { ExperimentNotebookPanel } from '../../components/ExperimentNotebookPanel';

const ExperimentNotebookBody: FC<PanelContext> = () => {
  return (
    <div style={{ padding: 14, height: '100%', overflowY: 'auto', boxSizing: 'border-box' }}>
      <ExperimentNotebookPanel />
    </div>
  );
};

pluginRegistry.register({
  id: 'experiment_notebook',
  title: 'Experiment Notebook',
  icon: 'BookOpen',
  category: 'experiments',
  resourceKinds: ['experiment', 'session', 'model'],
  defaultDock: 'center',
  Body: ExperimentNotebookBody,
});
