import { eventBus } from './eventBus';

/**
 * SelectionManager - Synchronizes active selections (layer, token, feature, neuron, session)
 * across all active visualization panels.
 */
class SelectionManager {
  constructor() {
    this.selection = {
      layer: 8,
      token: 'circuit',
      neuron: { layer: 8, index: 402 },
      feature: { id: 1402, label: 'Indirect Object' },
      session: 'sess_1',
    };
  }

  getSelection() {
    return { ...this.selection };
  }

  setLayer(layer) {
    this.selection.layer = layer;
    eventBus.emit('selection:changed', this.getSelection());
  }

  selectLayer(layer) {
    this.setLayer(layer);
  }

  setToken(token) {
    this.selection.token = token;
    eventBus.emit('selection:changed', this.getSelection());
  }

  selectToken(index, token) {
    this.setToken(token);
  }

  setNeuron(layer, index) {
    this.selection.neuron = { layer, index };
    eventBus.emit('selection:changed', this.getSelection());
  }

  selectNeuron(layer, index) {
    this.setNeuron(layer, index);
  }

  setFeature(featureId, label = '') {
    this.selection.feature = { id: featureId, label };
    eventBus.emit('selection:changed', this.getSelection());
  }

  selectFeature(featureId, label = '') {
    this.setFeature(featureId, label);
  }

  setSession(sessionId) {
    this.selection.session = sessionId;
    eventBus.emit('selection:changed', this.getSelection());
  }
}

export const selectionManager = new SelectionManager();
