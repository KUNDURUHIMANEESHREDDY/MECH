import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { PaperReproductionView } from '../../src/components/PaperReproductionView';
import { PluginSDKView } from '../../src/components/PluginSDKView';

describe('Sprint5ExplorerPanels', () => {
  it('renders PaperReproductionView with paper benchmarks', () => {
    render(<PaperReproductionView />);
    expect(screen.getByText(/Paper Reproduction/i)).toBeInTheDocument();
  });

  it('renders PluginSDKView component', () => {
    render(<PluginSDKView />);
    expect(screen.getByText(/Plugins/i)).toBeInTheDocument();
  });
});
