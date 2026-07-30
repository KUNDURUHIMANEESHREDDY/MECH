/**
 * Plugin Registry Service.
 */

export class PluginRegistryService {
  constructor() {
    this.plugins = [
      { id: 'plugin_1', name: 'SAE 3D Manifold Visualizer', author: 'VisualLab', version: '2.0.1', type: 'Visualization' },
      { id: 'plugin_2', name: 'Causal Linear Probe Toolkit', author: 'InterpTools', version: '1.4.0', type: 'Algorithm' },
    ];
  }

  listPlugins() {
    return [...this.plugins];
  }
}

export const pluginRegistryService = new PluginRegistryService();
