import { useWorkspaceStore } from '../stores/workspace';

export type DockZone = 'left' | 'center' | 'right' | 'bottom';

export class DockManager {
  private panelDockMap: Map<string, DockZone> = new Map([
    ['token_viewer', 'center'],
    ['attention_heatmap', 'center'],
    ['activation_heatmap', 'center'],
    ['neuron_panel', 'right'],
    ['layer_inspector', 'bottom'],
    ['prediction_inspector', 'bottom'],
    ['token_inspector', 'bottom'],
  ]);

  setDockZone(panelId: string, zone: DockZone) {
    this.panelDockMap.set(panelId, zone);
  }

  getDockZone(panelId: string): DockZone {
    return this.panelDockMap.get(panelId) || 'center';
  }

  togglePanel(panelId: string) {
    useWorkspaceStore.getState().togglePanel(panelId);
  }

  getVisiblePanels() {
    return useWorkspaceStore.getState().visiblePanels;
  }
}

export const dockManager = new DockManager();
