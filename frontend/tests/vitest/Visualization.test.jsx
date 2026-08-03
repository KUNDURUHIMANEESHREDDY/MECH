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
  it('renders NeuronSearch and filters by input', async () => {
    const neurons = [
      { label: 'Induction Head Key', neuron_index: 1, activation: 0.95, in_weight_l2: 0.1, out_weight_l2: 0.2 },
      { label: 'Name Mover Head', neuron_index: 2, activation: 0.8, in_weight_l2: 0.3, out_weight_l2: 0.4 }
    ];
    const api = {
      gpt2Neurons: async () => ({ status: 'ok', neurons, total_neurons: neurons.length })
    };
    render(<NeuronSearch api={api} />);
    await screen.findByText('Induction Head Key');
    const input = screen.getByPlaceholderText(/Search neuron ID/i);
    fireEvent.change(input, { target: { value: 'Induction' } });
    expect(screen.getByText('Induction Head Key')).toBeInTheDocument();
    expect(screen.queryByText('Name Mover Head')).not.toBeInTheDocument();
  });

  it('renders TimelinePanel and receives timeline events', () => {
    render(<TimelinePanel />);
    expect(screen.getByTestId('timeline-panel')).toBeInTheDocument();
  });
});
