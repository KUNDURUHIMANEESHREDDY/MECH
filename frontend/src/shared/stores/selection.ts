import { create } from 'zustand';
import { SelectionState, Resource } from '../types';
import { commandManager } from '../managers/commandManager';

export interface SelectionStore extends SelectionState {
  setSelectedResource: (resource: Resource | null) => void;
  setSelectedNeuron: (layer: number | null, neuron: number | null) => void;
  setSelectedHead: (layer: number | null, head: number | null) => void;
  setSelectedToken: (token: string | null) => void;
  setSelectedCircuit: (circuitId: string | null) => void;
  resetSelection: () => void;
}

export const useSelectionStore = create<SelectionStore>((set) => ({
  resource: null,
  layer: null,
  head: null,
  neuron: null,
  token: null,

  setSelectedResource: (resource) => {
    set({ resource });
    if (resource) {
      commandManager.publish('resource.selected', { resource });
    }
  },

  setSelectedNeuron: (layer, neuron) => {
    set({ layer, neuron });
    if (layer !== null && neuron !== null) {
      commandManager.publish('neuron.selected', { layer, head: 0, neuron });
    }
  },

  setSelectedHead: (layer, head) => {
    set({ layer, head });
  },

  setSelectedToken: (token) => {
    set({ token: token ? 0 : null }); // token stored as text in selection.resource for now
  },

  setSelectedCircuit: (circuitId) => {
    set({ resource: circuitId ? { id: circuitId, kind: 'circuit', label: circuitId } : null });
    if (circuitId) {
      commandManager.publish('circuit.highlighted', { circuitId });
    }
  },

  resetSelection: () => {
    set({ resource: null, layer: null, head: null, neuron: null, token: null });
  },
}));
