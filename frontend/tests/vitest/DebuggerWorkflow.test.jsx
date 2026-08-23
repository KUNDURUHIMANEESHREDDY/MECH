import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { NeuronSearch } from '../../src/components/NeuronSearch';
import { TransformerExplorer } from '../../src/components/TransformerExplorer';

describe('DebuggerWorkflow', () => {
  it('renders NeuronSearch panel with inputs and patching controls', () => {
    render(<NeuronSearch />);
    expect(screen.getByText('Neuron & Feature Search')).toBeInTheDocument();
    expect(screen.getByTestId('neuron-layer-input')).toBeInTheDocument();
    expect(screen.getByTestId('neuron-index-input')).toBeInTheDocument();
    expect(screen.getByTestId('apply-patch-btn')).toBeInTheDocument();
  });

  it('renders TransformerExplorer with layer select and MHA panel', () => {
    render(<TransformerExplorer />);
    expect(screen.getByText('Transformer Architecture Explorer')).toBeInTheDocument();
    expect(screen.getByTestId('mha-panel')).toBeInTheDocument();
    expect(screen.getByTestId('layer-select-btn-0')).toBeInTheDocument();
  });
});
