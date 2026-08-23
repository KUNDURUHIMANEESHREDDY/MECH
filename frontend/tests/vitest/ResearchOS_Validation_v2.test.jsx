import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PreregistrationModal } from '../../src/components/research/PreregistrationModal';
import { MechanismDiffView } from '../../src/components/research/MechanismDiffView';
import { GroundTruthBenchmarkView } from '../../src/components/research/GroundTruthBenchmarkView';
import { MechanismCriticPanel } from '../../src/components/research/MechanismCriticPanel';

describe('MECH Research Validation v2 - Preregistration, Benchmarks & Diff UI', () => {
  it('renders PreregistrationModal with protocol locking controls', () => {
    const handleClose = vi.fn();
    const handleLocked = vi.fn();
    render(<PreregistrationModal onClose={handleClose} onLocked={handleLocked} />);
    expect(screen.getByText(/Preregister & Lock Confirmatory Protocol/i)).toBeInTheDocument();
    expect(screen.getByText(/Empirical Prediction Statement/i)).toBeInTheDocument();
    expect(screen.getByText(/Negative Control Component/i)).toBeInTheDocument();
    expect(screen.getByText(/Lock Confirmatory Experiment/i)).toBeInTheDocument();
  });

  it('renders MechanismDiffView with version lineage and added components', () => {
    render(<MechanismDiffView />);
    expect(screen.getByText(/Mechanism Circuit Version Diff/i)).toBeInTheDocument();
    expect(screen.getByText(/Component Lineage Delta/i)).toBeInTheDocument();
    expect(screen.getByText(/Added Component: L10H0/i)).toBeInTheDocument();
    expect(screen.getByText(/Added Component: MLP_L8/i)).toBeInTheDocument();
  });

  it('renders GroundTruthBenchmarkView with literature circuit scorecards', () => {
    render(<GroundTruthBenchmarkView />);
    expect(screen.getByText(/Ground Truth vs Discovery Benchmark Scorecard/i)).toBeInTheDocument();
    expect(screen.getByText(/Indirect Object Identification \(IOI\) Circuit/i)).toBeInTheDocument();
    expect(screen.getByText(/In-Context Induction Head Circuit/i)).toBeInTheDocument();
    expect(screen.getByText(/Greater-Than Quantitative Reasoning Circuit/i)).toBeInTheDocument();
  });

  it('renders MechanismCriticPanel with audit and next experiment recommendation', () => {
    render(<MechanismCriticPanel />);
    expect(screen.getByText(/Mechanism Critic & Next-Best-Experiment Recommender/i)).toBeInTheDocument();
    expect(screen.getByText(/Recommended Next Best Experiment/i)).toBeInTheDocument();
    expect(screen.getByText(/Claim Epistemic Audit/i)).toBeInTheDocument();
  });
});
