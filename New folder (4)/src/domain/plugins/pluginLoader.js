/**
 * PluginLoader - Validates plugin manifests and registers extensions.
 */
export class PluginLoader {
  static validateManifest(manifest) {
    const required = ['id', 'name', 'version', 'minimumRuntimeVersion'];
    for (const key of required) {
      if (!manifest || !manifest[key]) {
        return { valid: false, error: `Missing manifest key: ${key}` };
      }
    }
    return { valid: true };
  }

  static loadPlugin(manifest) {
    const check = this.validateManifest(manifest);
    if (!check.valid) {
      throw new Error(check.error);
    }
    return {
      status: 'active',
      plugin_id: manifest.id,
      name: manifest.name,
      registered_panels: manifest.panels || [],
      registered_commands: manifest.commands || [],
    };
  }
}
