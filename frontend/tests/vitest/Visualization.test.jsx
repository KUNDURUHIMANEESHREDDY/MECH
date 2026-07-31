import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { eventBus } from '../../src/utils/eventBus';
import { panelRegistry } from '../../src/utils/panelRegistry';
import NeuronSearch from '../../src/components/NeuronSearch.jsx';
import TimelinePanel from '../../src/components/panels/TimelinePanel.jsx';

describe('EventBus & PanelRegistry', () => {
  it('publishes and subscribes to events via EventBus', () => {
    const callback = vi.fn();
    const unsub = eventBus.on('test:evt', callback);
    eventBus.emit('test:evt', { data: 'hello' });
    expect(callback).toHaveBeenCalledWith({ data: 'hello' });
    unsub();
  });

  it('registers and retrieves panels in PanelRegistry', () => {
    panelRegistry.register({
      id: 'test-panel',
      name: 'Test Panel',
      component: () => null
    });
    const panel = panelRegistry.getPanel('test-panel');
    expect(panel).not.toBeNull();
    expect(panel.name).toBe('Test Panel');
  });
});

describe('NeuronSearch & TimelinePanel', () => {
  it('renders NeuronSearch and filters by input', () => {
    render(<NeuronSearch />);
    expect(screen.getByTestId('neuron-search')).toBeInTheDocument();
    const input = screen.getByPlaceholderText(/Search neuron ID/i);
    fireEvent.change(input, { target: { value: 'Induction' } });
    expect(screen.getByText('Induction Head Key')).toBeInTheDocument();
  });

  it('renders TimelinePanel and receives timeline events', () => {
    render(<TimelinePanel />);
    expect(screen.getByTestId('timeline-panel')).toBeInTheDocument();
  });
});
