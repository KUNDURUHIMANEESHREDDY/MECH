import { AttentionMap, HeadData, LayerData } from '../types';

/**
 * Interprets raw attention map DTOs from the runtime into
 * the visualization layer's data structures.
 *
 * This is the **only** service that parses attention data.
 * Visualization components never touch raw tensors or runtime outputs.
 */
export function attentionMapsToLayers(
  maps: AttentionMap[],
  numLayers: number,
  numHeads: number,
  tokens: string[],
): LayerData[] {
  const layers: LayerData[] = [];

  for (let li = 0; li < numLayers; li++) {
    const heads: HeadData[] = [];
    for (let hi = 0; hi < numHeads; hi++) {
      const map = maps.find(m => m.layer === li && m.head === hi);
      const matrix = map?.matrix ?? [];
      heads.push({
        index: hi,
        attentionMatrix: matrix,
        neurons: [],
      });
    }
    layers.push({ index: li, heads });
  }

  return layers;
}
