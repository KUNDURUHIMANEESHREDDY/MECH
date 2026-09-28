import { describe, expect, it } from 'vitest';
import {
  LEGACY_ROUTE_IDS,
  ROUTES,
  getRouteById,
  isLegacyRoute,
} from '../../src/desktop/routeRegistry';

describe('desktop route registry', () => {
  it('keeps every hash route unique (34 slugs, 31 sidebar entries)', () => {
    expect(new Set(ROUTES.map(route => route.id)).size).toBe(ROUTES.length);
    expect(ROUTES).toHaveLength(34);
    expect(LEGACY_ROUTE_IDS).toHaveLength(33);
    expect(new Set(LEGACY_ROUTE_IDS).size).toBe(33);
    expect(ROUTES.filter(route => !route.aliasOf)).toHaveLength(31);
  });

  it('keeps alias slugs resolving to their canonical entry', () => {
    expect(getRouteById('gpt2explorer').aliasOf).toBe('explorer');
    expect(getRouteById('campaigns').aliasOf).toBe('workspace');
    expect(getRouteById('gpt2').aliasOf).toBe('explorer');
    expect(getRouteById('explorer').aliasOf).toBeUndefined();
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
