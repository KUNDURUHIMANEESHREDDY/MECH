import { render, screen } from '@testing-library/react';
import React from 'react';
import { describe, it, expect } from 'vitest';
import ResearchAnalyticsView from '../../src/components/ResearchAnalyticsView.jsx';

describe('ResearchAnalyticsView Component', () => {
  it('renders research analytics header and executive metrics', () => {
    render(<ResearchAnalyticsView />);
    expect(screen.getByText(/Multi-Campaign Meta-Learning Analytics/i)).toBeInTheDocument();
    expect(screen.getByText(/Campaigns Analyzed/i)).toBeInTheDocument();
    expect(screen.getByText(/Overall Success Rate/i)).toBeInTheDocument();
  });

  it('renders algorithm performance leaderboard table', () => {
    render(<ResearchAnalyticsView />);
    expect(screen.getByText(/Algorithm Performance Leaderboard/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Causal Scrubbing/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/ACDC Edge Pruning/i).length).toBeGreaterThan(0);
  });

  it('renders meta-learned optimal sequences section', () => {
    render(<ResearchAnalyticsView />);
    expect(screen.getByText(/Meta-Learned Optimal Experiment Sequences/i)).toBeInTheDocument();
    expect(screen.getByText(/IOI Standard Circuit Discovery/i)).toBeInTheDocument();
  });

  it('renders failure anti-patterns and compute efficiency meter', () => {
    render(<ResearchAnalyticsView />);
    expect(screen.getByText(/Learned Failure Anti-Patterns/i)).toBeInTheDocument();
    expect(screen.getByText(/Feature Universality before ACDC/i)).toBeInTheDocument();
    expect(screen.getByText(/Compute Efficiency Meter/i)).toBeInTheDocument();
  });
});
