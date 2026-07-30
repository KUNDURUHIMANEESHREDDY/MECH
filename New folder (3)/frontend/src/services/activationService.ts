import { NeuronActivation, NeuronData } from '../types';

/**
 * Interprets raw neuron activation DTOs from the runtime into
 * the visualization layer's data structures.
 *
 * Activations are MLP GELU outputs (true neuron activations),
 * NOT attention value vectors — those belong to attentionService.
 */
export function activationsToNeurons(
  activations: NeuronActivation[],
  numLayers: number,
): NeuronData[][] {
  const result: NeuronData[][] = Array.from({ length: numLayers }, () => []);

  for (const act of activations) {
    if (act.layer < numLayers) {
      result[act.layer].push({ index: act.index, activation: act.activation });
    }
  }

  return result;
}
