import { describe, it, expect } from 'vitest';
import { FrontendSDK } from '../../src/sdk/index.js';

describe('Sprint 3 Platform SDK & Frontend Tests', () => {
  it('registers panel plugin using FrontendSDK', () => {
    const reg = FrontendSDK.registerPanel(
      { id: 'custom_panel_1', name: 'Custom Probe Panel' },
      () => null
    );
    assert.strictEqual(reg.status, 'registered');
    assert.strictEqual(reg.panel_id, 'custom_panel_1');
  });

  it('registers algorithm plugin using FrontendSDK', () => {
    const reg = FrontendSDK.registerAlgorithm(
      { id: 'custom_algo_1', name: 'Custom Attribution Algorithm' },
      () => ({ score: 0.95 })
    );
    assert.strictEqual(reg.status, 'registered');
    assert.strictEqual(reg.algorithm_id, 'custom_algo_1');
  });
});
