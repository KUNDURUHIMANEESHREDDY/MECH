import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { TransformerExplorer } from '../../src/components/TransformerExplorer';

describe('TransformerVisualizer', () => {
  it('switches between Attention, MLP, and LayerNorm tabs', () => {
    render(<TransformerExplorer />);
    expect(screen.getByTestId('mha-panel')).toBeInTheDocument();

    const mlpTab = screen.getByTestId('tab-mlp');
    fireEvent.click(mlpTab);
    expect(screen.getByTestId('mlp-panel')).toBeInTheDocument();

    const lnTab = screen.getByTestId('tab-ln');
    fireEvent.click(lnTab);
    expect(screen.getByTestId('ln-panel')).toBeInTheDocument();
  });
});
