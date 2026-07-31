import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import CausalGraphEditorPanel from '../../src/components/panels/CausalGraphEditorPanel';
import TimeTravelDebuggerPanel from '../../src/components/panels/TimeTravelDebuggerPanel';
import LayerEvolutionPanel from '../../src/components/panels/LayerEvolutionPanel';
import FeatureGenealogyExplorerPanel from '../../src/components/panels/FeatureGenealogyExplorerPanel';
import ConfidenceHeatmapPanel from '../../src/components/panels/ConfidenceHeatmapPanel';
import DiscoveryComparisonPanel from '../../src/components/panels/DiscoveryComparisonPanel';
import InteractiveFiguresPanel from '../../src/components/panels/InteractiveFiguresPanel';

describe('AI 4 Scientific Analysis Visualization Panels', () => {
  it('renders CausalGraphEditorPanel and prunes node', () => {
    render(<CausalGraphEditorPanel />);
    expect(screen.getByTestId('causal-graph-editor-panel')).toBeInTheDocument();
    const pruneBtn = screen.getAllByText('Prune Node')[0];
    fireEvent.click(pruneBtn);
    expect(screen.getByText('Restore Node')).toBeInTheDocument();
  });

  it('renders TimeTravelDebuggerPanel and steps layer', () => {
    render(<TimeTravelDebuggerPanel />);
    expect(screen.getByTestId('time-travel-debugger-panel')).toBeInTheDocument();
    const stepBtn = screen.getByText('Step Forward');
    fireEvent.click(stepBtn);
    expect(screen.getByText('Layer 9')).toBeInTheDocument();
  });

  it('renders LayerEvolutionPanel', () => {
    render(<LayerEvolutionPanel />);
    expect(screen.getByTestId('layer-evolution-panel')).toBeInTheDocument();
  });

  it('renders FeatureGenealogyExplorerPanel', () => {
    render(<FeatureGenealogyExplorerPanel />);
    expect(screen.getByTestId('feature-genealogy-explorer-panel')).toBeInTheDocument();
  });

  it('renders ConfidenceHeatmapPanel', () => {
    render(<ConfidenceHeatmapPanel />);
    expect(screen.getByTestId('confidence-heatmap-panel')).toBeInTheDocument();
  });

  it('renders DiscoveryComparisonPanel', () => {
    render(<DiscoveryComparisonPanel />);
    expect(screen.getByTestId('discovery-comparison-panel')).toBeInTheDocument();
  });

  it('renders InteractiveFiguresPanel', () => {
    render(<InteractiveFiguresPanel />);
    expect(screen.getByTestId('interactive-figures-panel')).toBeInTheDocument();
  });
});
