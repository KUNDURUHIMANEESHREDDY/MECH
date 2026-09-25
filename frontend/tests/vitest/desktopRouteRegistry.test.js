import { describe, expect, it } from 'vitest';
import {
  LEGACY_ROUTE_IDS,
  ROUTES,
  getRouteById,
  isLegacyRoute,
} from '../../src/desktop/routeRegistry';

describe('desktop route registry', () => {
  it('keeps all 31 legacy hash routes plus the additive Society tool', () => {
    expect(new Set(ROUTES.map(route => route.id)).size).toBe(ROUTES.length);
    expect(ROUTES).toHaveLength(32);
    expect(LEGACY_ROUTE_IDS).toHaveLength(31);
    expect(new Set(LEGACY_ROUTE_IDS).size).toBe(31);
  });

  it('preserves canonical labels and deep-link routes', () => {
    expect(getRouteById('build').label).toBe('Build');
    expect(getRouteById('reasoning').label).toBe('Reasoning');
    expect(getRouteById('gpt2explorer').label).toBe('GPT-2 Neuron Explorer');
    expect(getRouteById('transformerExplorer').label).toBe('Transformer Explorer');
    expect(isLegacyRoute('gpt2explorer')).toBe(true);
    expect(isLegacyRoute('society')).toBe(false);
  });

  it('provides a lazy loader for every route', () => {
    for (const route of ROUTES) {
      expect(typeof route.load).toBe('function');
    }
  });

  it('returns null for unknown routes', () => {
    expect(getRouteById('not-a-route')).toBeNull();
  });
});
