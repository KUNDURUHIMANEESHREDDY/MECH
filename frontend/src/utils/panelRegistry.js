import { pluginRegistry } from '../services/panelRegistry';

export interface PanelDefinition {
  id: string;
  name: string;
  icon: any;
  desc: string;
  order: number;
  component: React.ComponentType;
}

export class PanelRegistry {
  private panels: Map<string, PanelDefinition> = new Map();

  register(panel: PanelDefinition) {
    this.panels.set(panel.id, panel);
  }

  get(id: string): PanelDefinition | undefined {
    return this.panels.get(id);
  }

  getPanels(): PanelDefinition[] {
    return Array.from(this.panels.values()).sort((a, b) => a.order - b.order);
  }

  list() {
    return this.getPanels();
  }
}

export const panelRegistry = new PanelRegistry();

// Register panels - these are the legacy panel components
// The actual panel components are imported in DockManager.jsx
