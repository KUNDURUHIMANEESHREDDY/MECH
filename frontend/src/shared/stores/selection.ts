import { create } from 'zustand';
import { SelectionState, Resource } from '../types';
import { commandManager } from '../managers/commandManager';

export interface ResearchComponent {
  name: string;
  layer?: number;
  head?: number;
  neuron?: number;
  featureId?: string;
  componentType?: 'head' | 'mlp' | 'feature' | 'residual' | 'token';
}

export interface SelectionStore extends SelectionState {
  researchComponent: ResearchComponent | null;
  selectedTokenIndex: number | null;
  setSelectedResource: (resource: Resource | null) => void;
  setSelectedNeuron: (layer: number | null, neuron: number | null) => void;
  setSelectedHead: (layer: number | null, head: number | null) => void;
  setSelectedToken: (token: string | null) => void;
  setSelectedCircuit: (circuitId: string | null) => void;
  setSelectedResearchComponent: (comp: ResearchComponent | null) => void;
  setSelectedTokenIndex: (idx: number | null) => void;
  resetSelection: () => void;
}

export const useSelectionStore = create<SelectionStore>((set) => ({
  resource: null,
  layer: null,
  head: null,
  neuron: null,
  token: null,
  researchComponent: null,
  selectedTokenIndex: null,

  setSelectedResource: (resource) => {
    set({ resource });
    if (resource) {
      commandManager.publish('resource.selected', { resource });
    }
  },

  setSelectedNeuron: (layer, neuron) => {
    set({
      layer,
      neuron,
      head: null,
      resource:
        layer !== null && neuron !== null
          ? { id: `L${layer}N${neuron}`, kind: 'neuron', label: `Neuron L${layer}.${neuron}` }
          : null,
    });
    if (layer !== null && neuron !== null) {
      commandManager.publish('neuron.selected', { layer, head: 0, neuron });
    }
  },

  setSelectedHead: (layer, head) => {
    set({
      layer,
      head,
      neuron: null,
      resource:
        layer !== null && head !== null
          ? { id: `L${layer}H${head}`, kind: 'head', label: `Head L${layer}.H${head}` }
          : null,
    });
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

  setSelectedResearchComponent: (comp) => {
    set({ researchComponent: comp });
    if (comp) {
      commandManager.publish('researchComponent.selected', { component: comp });
    }
  },

  setSelectedTokenIndex: (idx) => {
    set({ selectedTokenIndex: idx });
  },

  resetSelection: () => {
    set({ resource: null, layer: null, head: null, neuron: null, token: null, researchComponent: null, selectedTokenIndex: null });
  },
}));
