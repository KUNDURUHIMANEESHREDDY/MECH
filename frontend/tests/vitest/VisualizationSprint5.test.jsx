import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { TransformerNetworkPanel } from '../../src/components/panels/TransformerNetworkPanel';

describe('VisualizationSprint5', () => {
  it('renders TransformerNetworkPanel with full network layers', () => {
    render(<TransformerNetworkPanel />);
    expect(screen.getByText(/Run inference to inspect internals/i)).toBeInTheDocument();
  });
});
