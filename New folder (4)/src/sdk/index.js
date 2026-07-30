/**
 * Frontend Plugin SDK Entry Point.
 */
export class FrontendSDK {
  static registerPanel(manifest, component) {
    return {
      status: 'registered',
      panel_id: manifest.id,
      name: manifest.name,
      component,
    };
  }

  static registerAlgorithm(manifest, handler) {
    return {
      status: 'registered',
      algorithm_id: manifest.id,
      name: manifest.name,
      handler,
    };
  }
}
