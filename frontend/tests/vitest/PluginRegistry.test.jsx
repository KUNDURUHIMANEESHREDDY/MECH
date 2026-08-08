import React from 'react';
import { describe, it, expect, beforeEach } from 'vitest';
import { pluginRegistry } from '../../src/panel-system/pluginRegistry';

describe('PluginRegistry', () => {
  beforeEach(() => {
    pluginRegistry.clear();
  });

  it('registers and retrieves panel plugins', () => {
    const dummyPlugin = {
      id: 'dummy_panel',
      title: 'Dummy Panel',
      icon: 'Square',
      category: 'testing',
      resourceKinds: ['model', 'neuron'],
      defaultDock: 'center',
      Body: () => <div>Dummy Content</div>,
      commands: [{ id: 'dummy.cmd', label: 'Run Dummy', handler: () => {} }],
    };

    pluginRegistry.register(dummyPlugin);

    expect(pluginRegistry.get('dummy_panel')).toBe(dummyPlugin);
    expect(pluginRegistry.list()).toHaveLength(1);
    expect(pluginRegistry.getByResourceKind('neuron')).toContain(dummyPlugin);
    expect(pluginRegistry.getByCategory('testing')).toContain(dummyPlugin);
  });
});
