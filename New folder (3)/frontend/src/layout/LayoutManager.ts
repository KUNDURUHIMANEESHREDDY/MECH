export interface LayoutConfig {
  visiblePanels: Record<string, boolean>;
  dockPositions: Record<string, 'left' | 'center' | 'right' | 'bottom'>;
}

const STORAGE_KEY = 'model_explorer_layout_v1';

export class LayoutManager {
  private config: LayoutConfig;

  constructor() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        this.config = JSON.parse(saved);
      } catch {
        this.config = this.getDefaults();
      }
    } else {
      this.config = this.getDefaults();
    }
  }

  private getDefaults(): LayoutConfig {
    return {
      visiblePanels: {
        token_viewer: true,
        attention_heatmap: true,
        activation_heatmap: true,
        neuron_panel: true,
        layer_inspector: true,
        prediction_inspector: true,
        token_inspector: true,
      },
      dockPositions: {
        token_viewer: 'center',
        attention_heatmap: 'center',
        activation_heatmap: 'center',
        neuron_panel: 'right',
        layer_inspector: 'bottom',
        prediction_inspector: 'bottom',
        token_inspector: 'bottom',
      },
    };
  }

  getConfig(): LayoutConfig {
    return this.config;
  }

  togglePanel(panelId: string): boolean {
    const current = !!this.config.visiblePanels[panelId];
    this.config.visiblePanels[panelId] = !current;
    this.save();
    return !current;
  }

  save() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(this.config));
  }
}

export const layoutManager = new LayoutManager();
