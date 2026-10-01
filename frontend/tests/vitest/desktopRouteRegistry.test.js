import { describe, expect, it } from 'vitest';
import {
  LEGACY_ROUTE_IDS,
  ROUTES,
  getRouteById,
  isLegacyRoute,
} from '../../src/desktop/routeRegistry';

// Every deep-link route that shipped before Society. This list is the actual
// contract: a previously published hash link must keep resolving.
//
// The test used to assert `ROUTES.length === 32` and `LEGACY_ROUTE_IDS.length
// === 31`. LEGACY_ROUTE_IDS is derived from ROUTES by filtering route.legacy,
// so those were counts of a derived value that broke whenever an additive
// route was added -- which is what happened (33 legacy + society = 34 today).
// A count cannot express backward compatibility; membership can.
const LEGACY_IDS = [
  'explorer',
  'gpt2',
  'gpt2explorer',
  'transformer',
  'transformerExplorer',
  'network',
  'steering',
  'workspace',
  'models',
  'prompts',
  'debugger',
  'experiments',
  'sessions',
  'reports',
  'settings',
  'logging',
  'build',
  'neuralexplorer',
  'benchmark',
  'benchmarksuite',
  'knowledgegraph',
  'circuitexplorer',
  'reasoning',
  'evidencefusion',
  'campaigns',
  'analytics',
  'health',
  'plugins',
  'notebook',
  'labnotebook',
  'reproduction',
  'projects',
  'recent',
];

describe('desktop route registry', () => {
  it('has unique route ids', () => {
    expect(new Set(ROUTES.map((route) => route.id)).size).toBe(ROUTES.length);
  });

  it('keeps every legacy deep link resolving and marked legacy', () => {
    const missing = LEGACY_IDS.filter(
      (id) => !ROUTES.some((route) => route.id === id),
    );
    expect(missing, `routes removed: ${missing.join(', ')}`).toEqual([]);

    const noLongerLegacy = LEGACY_IDS.filter((id) => !isLegacyRoute(id));
    expect(
      noLongerLegacy,
      `routes stopped being legacy, which breaks existing deep links: ${noLongerLegacy.join(', ')}`,
    ).toEqual([]);
  });

  it('exposes the legacy list in registration order without duplicates', () => {
    expect(new Set(LEGACY_ROUTE_IDS).size).toBe(LEGACY_ROUTE_IDS.length);
    for (const id of LEGACY_IDS) {
      expect(LEGACY_ROUTE_IDS).toContain(id);
    }
  });

  it('treats Society as additive, not a legacy replacement', () => {
    expect(getRouteById('society')).not.toBeNull();
    expect(isLegacyRoute('society')).toBe(false);
    expect(LEGACY_ROUTE_IDS).not.toContain('society');
  });

  it('preserves canonical labels', () => {
    expect(getRouteById('build').label).toBe('Build');
    expect(getRouteById('reasoning').label).toBe('Reasoning');
    expect(getRouteById('gpt2explorer').label).toBe('GPT-2 Neuron Explorer');
    expect(getRouteById('transformerExplorer').label).toBe('Transformer Explorer');
  });

  it('provides a lazy loader for every route', () => {
    for (const route of ROUTES) {
      expect(typeof route.load, `route ${route.id} has no loader`).toBe('function');
    }
  });

  it('returns null for unknown routes', () => {
    expect(getRouteById('not-a-route')).toBeNull();
  });
});
