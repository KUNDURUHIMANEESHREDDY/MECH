import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import CampaignWorkspaceView from '../../src/components/CampaignWorkspaceView.jsx';

describe('CampaignWorkspaceView Component', () => {
  it('renders campaign workspace title and metrics header', () => {
    render(<CampaignWorkspaceView />);
    expect(screen.getByText(/Scientist's Campaign Workspace/i)).toBeInTheDocument();
    expect(screen.getAllByText(/IOI Circuit Discovery & Cross-Model Validation/i).length).toBeGreaterThan(0);
  });

  it('renders Bayesian Belief Evolution panel', () => {
    render(<CampaignWorkspaceView />);
    const beliefTab = screen.getByText(/^Bayesian Belief Evolution$/i);
    fireEvent.click(beliefTab);
    expect(screen.getByText(/Mathematically Traceable Bayesian Belief Evolution/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Attribution Patching/i).length).toBeGreaterThan(0);
  });

  it('renders completed and failed experiment cards in matrix panel', () => {
    render(<CampaignWorkspaceView />);
    const expTab = screen.getByText(/^Executed Matrix$/i);
    fireEvent.click(expTab);
    expect(screen.getAllByText(/Attribution Patching/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/ACDC Edge Pruning/i).length).toBeGreaterThan(0);
  });

  it('switches active campaign when tab button is clicked', () => {
    render(<CampaignWorkspaceView />);
    const inductionTab = screen.getByText(/Induction Head Sequence Repeater Study/i);
    fireEvent.click(inductionTab);
    expect(screen.getByText(/Isolate prefix-matching and token-copying induction heads/i)).toBeInTheDocument();

    const expTab = screen.getByText(/^Executed Matrix$/i);
    fireEvent.click(expTab);
    expect(screen.getByText(/Naive Random Resampling/i)).toBeInTheDocument();
  });
});
