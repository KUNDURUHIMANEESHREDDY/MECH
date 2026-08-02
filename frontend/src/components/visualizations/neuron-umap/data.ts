import { HeadData, LayerData, NeuronData } from '../../types';
import { NeuronPoint } from './types';

export function weightStatsOf(values: number[] | undefined): { mean: number; std: number; min: number; max: number } | undefined {
  if (!values || values.length === 0) return undefined;
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const mean = sum / values.length;
  let sq = 0;
  for (const v of values) sq += (v - mean) * (v - mean);
  return { mean, std: values.length > 1 ? Math.sqrt(sq / (values.length - 1)) : 0, min, max };
}

export function idForHeadNeuron(layer: number, head: number, neuron: number): string {
  return `L${layer}H${head}N${neuron}`;
}

export function idForLayerNeuron(layer: number, neuron: number): string {
  return `L${layer}N${neuron}`;
}

export function parseNeuronId(id: string | null): { layer: number; head?: number; neuron: number } | null {
  if (!id) return null;
  const headMatch = /^L(\d+)H(\d+)N(\d+)$/.exec(id);
  if (headMatch) {
    return { layer: Number(headMatch[1]), head: Number(headMatch[2]), neuron: Number(headMatch[3]) };
  }
  const layerMatch = /^L(\d+)N(\d+)$/.exec(id);
  if (layerMatch) {
    return { layer: Number(layerMatch[1]), neuron: Number(layerMatch[2]) };
  }
  return null;
}

export function buildNeuronPoints(layers: LayerData[], tokens?: string[]): NeuronPoint[] {
  const points: NeuronPoint[] = [];
  for (const layer of layers) {
    for (const head of layer.heads) {
      for (const n of head.neurons) {
        points.push(buildHeadNeuronPoint(layer.index, head.index, n, tokens));
      }
    }
  }
  return points;
}

export function buildHeadNeuronPoint(layer: number, head: number, n: NeuronData, tokens?: string[]): NeuronPoint {
  const tokenActivations = n.tokenActivations && n.tokenActivations.length > 0 ? n.tokenActivations : undefined;
  return {
    id: idForHeadNeuron(layer, head, n.index),
    neuronIndex: n.index,
    layer,
    head,
    activation: n.activation,
    embedding: tokenActivations && tokenActivations.length > 0 ? tokenActivations : [n.activation],
    tokenActivations,
    tokens: tokens && tokenActivations ? tokens : undefined,
    weightStats: weightStatsOf(tokenActivations),
  };
}

export function buildLayerNeuronPoints(
  neurons: Array<{
    neuron_index: number;
    activation: number;
    in_weight_l2?: number | null;
    out_weight_l2?: number | null;
    top_token?: string | null;
  }>,
  layer: number,
): NeuronPoint[] {
  const points: NeuronPoint[] = [];
  for (const n of neurons) {
    const inW = typeof n.in_weight_l2 === 'number' ? n.in_weight_l2 : null;
    const outW = typeof n.out_weight_l2 === 'number' ? n.out_weight_l2 : null;
    const embedding = inW !== null ? [n.activation, inW] : [n.activation];
    const weightValues =
      inW !== null && outW !== null ? [inW, outW] : inW !== null ? [inW] : outW !== null ? [outW] : undefined;
    points.push({
      id: idForLayerNeuron(layer, n.neuron_index),
      neuronIndex: n.neuron_index,
      layer,
      activation: n.activation,
      embedding,
      weightStats: weightStatsOf(weightValues),
      topToken: n.top_token ?? undefined,
    });
  }
  return points;
}

export function buildTokenActivationPoints(
  tokenActivations: number[],
  neuronIndex: number,
  layer: number,
  tokens?: string[],
): NeuronPoint {
  return {
    id: idForLayerNeuron(layer, neuronIndex),
    neuronIndex,
    layer,
    activation: tokenActivations.reduce((a, b) => a + b, 0) / (tokenActivations.length || 1),
    embedding: tokenActivations.length > 0 ? tokenActivations : [0],
    tokenActivations: tokenActivations.length > 0 ? tokenActivations : undefined,
    tokens: tokens && tokenActivations.length > 0 ? tokens : undefined,
    weightStats: weightStatsOf(tokenActivations),
  };
}

export function headIndexOf(heads: HeadData[] | undefined): number {
  return heads && heads.length > 0 ? heads[0].index : 0;
}
