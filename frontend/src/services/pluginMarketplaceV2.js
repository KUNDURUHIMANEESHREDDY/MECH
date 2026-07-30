/**
 * Plugin Marketplace v2 Service.
 */

export class PluginMarketplaceV2 {
  constructor() {
    this.plugins = [
      { id: 'plugin_sae_viz', name: 'SAE 3D Manifold Visualizer', author: 'VisualLab', version: '2.0.1' },
      { id: 'plugin_causal_probe', name: 'Causal Linear Probe Toolkit', author: 'InterpTools', version: '1.4.0' }
    ];
  }

  listPlugins() {
    return [...this.plugins];
  }
}
