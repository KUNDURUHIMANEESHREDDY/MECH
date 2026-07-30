/**
 * Extension Marketplace Service (Sprint 3 AI 5)
 * Plugin discovery, validation, and installation manager.
 */

export class ExtensionMarketplaceService {
  constructor() {
    this.installedPlugins = [
      { id: "sae_probes_v1", name: "SAE Probes Extension", version: "1.0.0", status: "installed" },
    ];
  }

  listAvailablePlugins() {
    return [
      { id: "sae_probes_v1", name: "SAE Probes Extension", author: "Anthropic Research", version: "1.0.0" },
      { id: "transformer_lens_adapter", name: "TransformerLens Bridge", author: "Neel Nanda Lab", version: "2.1.0" },
      { id: "pythia_suite", name: "Pythia Suite Inspectors", author: "EleutherAI", version: "1.4.0" },
    ];
  }

  installPlugin(pluginId) {
    const available = this.listAvailablePlugins().find((p) => p.id === pluginId);
    if (!available) throw new Error(`Plugin ${pluginId} not found in marketplace`);
    
    if (!this.installedPlugins.some((p) => p.id === pluginId)) {
      this.installedPlugins.push({ ...available, status: "installed" });
    }
    return { status: "installed", plugin: available };
  }

  listInstalledPlugins() {
    return [...this.installedPlugins];
  }
}

export const extensionMarketplaceService = new ExtensionMarketplaceService();
