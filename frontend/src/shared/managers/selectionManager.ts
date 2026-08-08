import { useSelectionStore } from '../stores/selection';
import { Resource } from '../types';

export class SelectionManager {
  selectResource(resource: Resource | null) {
    useSelectionStore.getState().setSelectedResource(resource);
  }

  selectNeuron(layer: number | null, neuron: number | null) {
    useSelectionStore.getState().setSelectedNeuron(layer, neuron);
  }

  selectHead(layer: number | null, head: number | null) {
    useSelectionStore.getState().setSelectedHead(layer, head);
  }

  selectToken(token: string | null) {
    useSelectionStore.getState().setSelectedToken(token);
  }

  selectCircuit(circuitId: string | null) {
    useSelectionStore.getState().setSelectedCircuit(circuitId);
  }

  getSelection() {
    return useSelectionStore.getState();
  }

  reset() {
    useSelectionStore.getState().resetSelection();
  }
}

export const selectionManager = new SelectionManager();
