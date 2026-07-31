import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import EvidenceFusionView from '../../src/components/EvidenceFusionView.jsx';

describe('EvidenceFusionView Component', () => {
  it('renders evidence fusion header and claim title', () => {
    render(<EvidenceFusionView />);
    expect(screen.getByText(/Mechanism Claim Registry/i)).toBeInTheDocument();
    expect(screen.getAllByText(/IOI Name Mover Circuit/i).length).toBeGreaterThan(0);
  });

  it('renders per-algorithm evidence table entries', () => {
    render(<EvidenceFusionView />);
    expect(screen.getAllByText(/Attribution Patching/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/ACDC/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Path Patching/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Causal Scrubbing/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Feature Universality/i).length).toBeGreaterThan(0);
  });

  it('switches claim when tab button is clicked', () => {
    render(<EvidenceFusionView />);
    const button = screen.getAllByText(/Induction Head Sequence Repeater/i)[0];
    fireEvent.click(button);
    expect(screen.getByText(/Previous-token head L4H2 attends to token K-1/i)).toBeInTheDocument();
  });

  it('navigates to Circuit Explorer when link is clicked', () => {
    const onNavigate = vi.fn();
    render(<EvidenceFusionView onNavigate={onNavigate} />);
    const link = screen.getByText(/Circuit Explorer/i);
    fireEvent.click(link);
    expect(onNavigate).toHaveBeenCalledWith('circuitexplorer');
  });
});
