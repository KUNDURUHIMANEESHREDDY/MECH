import { notebookStore } from '../notebook/notebookStore';
import { layoutManager } from './layoutManager';

/**
 * WorkspaceExporter & Importer - Packages complete research workspaces for round-trip export & import.
 */
export class WorkspaceBundle {
  static exportBundle() {
    return {
      version: '2.0.0',
      exported_at: new Date().toISOString(),
      layout: layoutManager.getLayout(),
      notebook: notebookStore.exportNotebook(),
      metadata: {
        // Never asserted: no live model/dataset identity is available in
        // this module. Previously hardcoded to GPT-2 Small / IOI Benchmark.
        active_model: 'unconfirmed',
        dataset: 'unconfirmed',
      },
    };
  }

  static importBundle(bundle) {
    if (!bundle || !bundle.layout) {
      throw new Error('Invalid workspace bundle format');
    }
    layoutManager.restoreLayout(bundle.layout);
    return {
      success: true,
      restored_at: new Date().toISOString(),
      bundle,
    };
  }
}
