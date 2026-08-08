import { pluginRegistry } from '../../panel-system/pluginRegistry';
import { useWorkspaceStore } from '../stores/workspace';

export class PanelManager {
  openPanel(panelId: string, dock?: 'left' | 'center' | 'right' | 'bottom') {
    const plugin = pluginRegistry.get(panelId);
    if (!plugin) {
      console.warn(`[PanelManager] Plugin not found for id: ${panelId}`);
    }
    useWorkspaceStore.getState().openPanel(panelId);
  }

  closePanel(panelId: string) {
    useWorkspaceStore.getState().closePanel(panelId);
  }

  togglePanel(panelId: string) {
    useWorkspaceStore.getState().togglePanel(panelId);
  }

  isPanelVisible(panelId: string): boolean {
    return !!useWorkspaceStore.getState().visiblePanels[panelId];
  }

  getRegisteredPlugins() {
    return pluginRegistry.list();
  }
}

export const panelManager = new PanelManager();
