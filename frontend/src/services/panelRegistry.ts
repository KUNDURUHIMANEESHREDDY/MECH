import type { LucideIcon } from 'lucide-react';
import { Brain, Flame, Layers, Search, Telescope, Type, Workflow, Zap } from 'lucide-react';

export interface PanelDefinition {
  id: string;
  title: string;
  icon: LucideIcon;
  defaultDock: 'left' | 'center' | 'right' | 'bottom';
  commands: string[];
}

export class PanelRegistry {
  private panels: Map<string, PanelDefinition> = new Map();

  register(panel: PanelDefinition) {
    this.panels.set(panel.id, panel);
  }

  get(id: string): PanelDefinition | undefined {
    return this.panels.get(id);
  }

  list(): PanelDefinition[] {
    return Array.from(this.panels.values());
  }
}

export const panelRegistry = new PanelRegistry();

// Register Sprint 1 core panels
panelRegistry.register({
  id: 'token_viewer',
  title: 'Token Viewer',
  icon: Type,
  defaultDock: 'center',
  commands: ['toggle_token_viewer', 'view_tokens'],
});

panelRegistry.register({
  id: 'attention_heatmap',
  title: 'Attention Heatmap',
  icon: Flame,
  defaultDock: 'center',
  commands: ['toggle_attention_heatmap', 'view_attention'],
});

panelRegistry.register({
  id: 'activation_heatmap',
  title: 'Activation Heatmap',
  icon: Zap,
  defaultDock: 'center',
  commands: ['toggle_activation_heatmap', 'view_activations'],
});

panelRegistry.register({
  id: 'neuron_panel',
  title: 'Neuron Inspector',
  icon: Brain,
  defaultDock: 'right',
  commands: ['toggle_neuron_panel', 'inspect_neuron'],
});

panelRegistry.register({
  id: 'layer_inspector',
  title: 'Layer Inspector',
  icon: Layers,
  defaultDock: 'bottom',
  commands: ['toggle_layer_inspector', 'inspect_layer'],
});

panelRegistry.register({
  id: 'prediction_inspector',
  title: 'Prediction Inspector',
  icon: Telescope,
  defaultDock: 'bottom',
  commands: ['toggle_prediction_inspector', 'inspect_predictions', 'logit_lens'],
});

panelRegistry.register({
  id: 'token_inspector',
  title: 'Token Inspector',
  icon: Search,
  defaultDock: 'bottom',
  commands: ['toggle_token_inspector', 'inspect_token'],
});

panelRegistry.register({
  id: 'society',
  title: 'Society Runs',
  icon: Workflow,
  defaultDock: 'center',
  commands: ['toggle_society', 'run_society'],
});
