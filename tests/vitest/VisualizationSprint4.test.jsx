import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import ResearchGraphPanel from '../../src/components/panels/ResearchGraphPanel';
import CircuitAnimationPanel from '../../src/components/panels/CircuitAnimationPanel';
import KnowledgeGraphViewerPanel from '../../src/components/panels/KnowledgeGraphViewerPanel';
import VR3DNetworkViewerPanel from '../../src/components/panels/VR3DNetworkViewerPanel';
import PublicationDashboardPanel from '../../src/components/panels/PublicationDashboardPanel';
import PresentationModePanel from '../../src/components/panels/PresentationModePanel';
import LiveCollaborationPanel from '../../src/components/panels/LiveCollaborationPanel';
import { visualizationEngine } from '../../src/services/visualizationEngine';

describe('Sprint 4 Scientific Visualization Panels', () => {
  it('renders ResearchGraphPanel and toggles layout', () => {
    render(<ResearchGraphPanel />);
    expect(screen.getByTestId('research-graph-panel')).toBeInTheDocument();
    const btn = screen.getByText('Force');
    fireEvent.click(btn);
    expect(visualizationEngine.layoutMode).toBe('Force');
  });

  it('renders CircuitAnimationPanel and controls playback', () => {
    render(<CircuitAnimationPanel />);
    expect(screen.getByTestId('circuit-animation-panel')).toBeInTheDocument();
    const playBtn = screen.getByText('▶ Play');
    fireEvent.click(playBtn);
    expect(visualizationEngine.playbackState.isPlaying).toBe(true);
  });

  it('renders KnowledgeGraphViewerPanel and filters types', () => {
    render(<KnowledgeGraphViewerPanel />);
    expect(screen.getByTestId('knowledge-graph-panel')).toBeInTheDocument();
    const neuronBtn = screen.getByRole('button', { name: 'Neuron' });
    fireEvent.click(neuronBtn);
    expect(screen.getByText('Neuron L8_N402')).toBeInTheDocument();
  });

  it('renders VR3DNetworkViewerPanel', () => {
    render(<VR3DNetworkViewerPanel />);
    expect(screen.getByTestId('vr-3d-panel')).toBeInTheDocument();
  });

  it('renders PublicationDashboardPanel', () => {
    render(<PublicationDashboardPanel />);
    expect(screen.getByTestId('publication-dashboard-panel')).toBeInTheDocument();
  });

  it('renders PresentationModePanel', () => {
    render(<PresentationModePanel />);
    expect(screen.getByTestId('presentation-mode-panel')).toBeInTheDocument();
  });

  it('renders LiveCollaborationPanel', () => {
    render(<LiveCollaborationPanel />);
    expect(screen.getByTestId('live-collaboration-panel')).toBeInTheDocument();
  });
});
