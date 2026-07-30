import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import InferenceTimeline from '../../src/components/panels/InferenceTimeline.jsx';
import SAEFeaturePanel from '../../src/components/panels/SAEFeaturePanel.jsx';
import CircuitGraphPanel from '../../src/components/panels/CircuitGraphPanel.jsx';
import ResidualStreamViewer from '../../src/components/panels/ResidualStreamViewer.jsx';
import LogitLensViewer from '../../src/components/panels/LogitLensViewer.jsx';
import LayerComparisonPanel from '../../src/components/panels/LayerComparisonPanel.jsx';
import ModelComparisonPanel from '../../src/components/panels/ModelComparisonPanel.jsx';
import { selectionManager } from '../../src/utils/selectionManager.js';

describe('Sprint 2 Visualization Panels', () => {
  it('renders InferenceTimeline and handles stepping', () => {
    render(<InferenceTimeline />);
    expect(screen.getByTestId('inference-timeline')).toBeInTheDocument();
    const stepBtn = screen.getByText(/Step Layer/i);
    fireEvent.click(stepBtn);
    expect(selectionManager.getSelection().layer).toBe(1);
  });

  it('renders SAEFeaturePanel', async () => {
    render(<SAEFeaturePanel />);
    expect(screen.getByTestId('sae-feature-panel')).toBeInTheDocument();
  });

  it('renders CircuitGraphPanel and handles node selection', () => {
    render(<CircuitGraphPanel />);
    expect(screen.getByTestId('circuit-graph-panel')).toBeInTheDocument();
  });

  it('renders ResidualStreamViewer', () => {
    render(<ResidualStreamViewer />);
    expect(screen.getByTestId('residual-stream-panel')).toBeInTheDocument();
  });

  it('renders LogitLensViewer and toggles lens method', () => {
    render(<LogitLensViewer />);
    expect(screen.getByTestId('logit-lens-panel')).toBeInTheDocument();
    const tunedBtn = screen.getByText('Tuned Lens');
    fireEvent.click(tunedBtn);
  });

  it('renders LayerComparisonPanel', () => {
    render(<LayerComparisonPanel />);
    expect(screen.getByTestId('layer-comparison-panel')).toBeInTheDocument();
  });

  it('renders ModelComparisonPanel', () => {
    render(<ModelComparisonPanel />);
    expect(screen.getByTestId('model-comparison-panel')).toBeInTheDocument();
  });
});
