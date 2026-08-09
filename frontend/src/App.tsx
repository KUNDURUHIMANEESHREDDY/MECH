import React from 'react';
import { useModel } from './shared/hooks/useModel';
import { Shell } from './app/Shell';
import { useUIStore } from './shared/stores/ui';
import { useAppStore } from './store/useAppStore';
import { api } from './services/api';

export default function App() {
  const { state: model, listModels, load } = useModel();
  const toggleCommandPalette = useUIStore((s) => s.toggleCommandPalette);
  const setPythonStatus = useAppStore((s) => s.setPythonStatus);

  React.useEffect(() => {
    listModels();
    api
      .pythonPing()
      .then(() => setPythonStatus(true))
      .catch(() => setPythonStatus(false));
  }, []);

  const handleLoadModel = (name: string) => {
    load(name);
  };

  const handleRunPrompt = () => {
    // In the canvas shell, prompts are opened as resources rather than run inline.
    // Actual inference happens inside panel plugins.
  };

  return (
    <Shell
      pythonStatus={!model.error}
      onOpenCommandPalette={toggleCommandPalette}
      onLoadModel={handleLoadModel}
      onRunPrompt={handleRunPrompt}
    >
      {/* Panel children are rendered by DockManager via PluginRegistry,
          not passed as children here. */}
    </Shell>
  );
}
