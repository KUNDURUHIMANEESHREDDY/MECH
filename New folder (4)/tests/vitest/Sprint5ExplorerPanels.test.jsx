import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import KnowledgeUniversePanel from '../../src/components/panels/KnowledgeUniversePanel';
import MechanismSimulatorPanel from '../../src/components/panels/MechanismSimulatorPanel';
import ResearchReplayPanel from '../../src/components/panels/ResearchReplayPanel';
import DiscoveryTimelinePanel from '../../src/components/panels/DiscoveryTimelinePanel';
import ConferenceModePanel from '../../src/components/panels/ConferenceModePanel';
import ImmersiveCollabPanel from '../../src/components/panels/ImmersiveCollabPanel';
import PublicationStudioPanel from '../../src/components/panels/PublicationStudioPanel';

describe('Sprint 5 Knowledge Explorer Panels', () => {
  it('renders KnowledgeUniversePanel', () => {
    render(<KnowledgeUniversePanel />);
    expect(screen.getByTestId('knowledge-universe-panel')).toBeInTheDocument();
  });

  it('renders MechanismSimulatorPanel', () => {
    render(<MechanismSimulatorPanel />);
    expect(screen.getByTestId('mechanism-simulator-panel')).toBeInTheDocument();
  });

  it('renders ResearchReplayPanel', () => {
    render(<ResearchReplayPanel />);
    expect(screen.getByTestId('research-replay-panel')).toBeInTheDocument();
  });

  it('renders DiscoveryTimelinePanel', () => {
    render(<DiscoveryTimelinePanel />);
    expect(screen.getByTestId('discovery-timeline-panel')).toBeInTheDocument();
  });

  it('renders ConferenceModePanel', () => {
    render(<ConferenceModePanel />);
    expect(screen.getByTestId('conference-mode-panel')).toBeInTheDocument();
  });

  it('renders ImmersiveCollabPanel', () => {
    render(<ImmersiveCollabPanel />);
    expect(screen.getByTestId('immersive-collab-panel')).toBeInTheDocument();
  });

  it('renders PublicationStudioPanel', () => {
    render(<PublicationStudioPanel />);
    expect(screen.getByTestId('publication-studio-panel')).toBeInTheDocument();
  });
});
