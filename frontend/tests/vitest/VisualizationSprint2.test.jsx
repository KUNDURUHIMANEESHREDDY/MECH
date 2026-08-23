import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { Gpt2NeuronExplorerPanel } from '../../src/components/panels/Gpt2NeuronExplorerPanel';

describe('VisualizationSprint2', () => {
  it('renders Gpt2NeuronExplorerPanel with neuron activation visualizations', () => {
    render(<Gpt2NeuronExplorerPanel />);
    expect(screen.getByText(/Probe Sequence/i)).toBeInTheDocument();
  });
});
