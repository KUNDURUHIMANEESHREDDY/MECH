import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { GlobalQuickSearch } from '../../src/components/research/GlobalQuickSearch';
import { MethodologyDrawer } from '../../src/components/research/MethodologyDrawer';
import { MechanismGraphCenterpiece } from '../../src/components/research/MechanismGraphCenterpiece';
import { ResearchQueueView } from '../../src/components/research/ResearchQueueView';
import { ComparisonWorkspaceView } from '../../src/components/research/ComparisonWorkspaceView';
import { WelcomeOnboardingModal } from '../../src/components/research/WelcomeOnboardingModal';
import { UniversalInspector } from '../../src/components/research/UniversalInspector';

describe('MECH v2.0 Productization — Research OS Component Suite', () => {
  it('renders GlobalQuickSearch and filters components by query', () => {
    const handleSelect = vi.fn();
    const handleClose = vi.fn();

    render(<GlobalQuickSearch isOpen={true} onClose={handleClose} onSelect={handleSelect} />);

    expect(screen.getByPlaceholderText(/Search components/i)).toBeInTheDocument();
    expect(screen.getByText(/L9H9 \(Name Mover Head\)/i)).toBeInTheDocument();

    const input = screen.getByPlaceholderText(/Search components/i);
    fireEvent.change(input, { target: { value: 'Induction' } });

    expect(screen.getByText(/L5H5 \(Induction Head\)/i)).toBeInTheDocument();
  });

  it('renders MethodologyDrawer with detailed experimental parameters', () => {
    const handleClose = vi.fn();
    render(<MethodologyDrawer isOpen={true} onClose={handleClose} />);

    expect(screen.getByText(/Experimental Methodology/i)).toBeInTheDocument();
    expect(screen.getByText(/Model & Tokenizer/i)).toBeInTheDocument();
    expect(screen.getByText(/Dataset & Prompt Construction/i)).toBeInTheDocument();
    expect(screen.getByText(/Intervention & Negative Control/i)).toBeInTheDocument();
    expect(screen.getByText(/Metrics & Estimation/i)).toBeInTheDocument();
    expect(screen.getByText(/Isolates target-specific causal logit shift/i)).toBeInTheDocument();
  });

  it('renders MechanismGraphCenterpiece with inspectable causal edges and nodes', () => {
    render(<MechanismGraphCenterpiece />);

    expect(screen.getByText(/Mechanistic Circuit Graph/i)).toBeInTheDocument();
    expect(screen.getByText(/L8H1/i)).toBeInTheDocument();
    expect(screen.getByText(/L9H9/i)).toBeInTheDocument();
    expect(screen.getByText(/RESIDUAL/i)).toBeInTheDocument();
  });

  it('renders ResearchQueueView and handles Run All Queued Experiments', async () => {
    render(<ResearchQueueView />);

    expect(screen.getByText(/What Should I Test Next\? — Research Queue/i)).toBeInTheDocument();
    expect(screen.getByText(/Ablate MLP_L8 under Matched IOI Conditions/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Outcome A:/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Outcome B:/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Outcome C:/i).length).toBeGreaterThan(0);

    const runBtn = screen.getByText(/Run All Queued Experiments/i);
    fireEvent.click(runBtn);
    expect(screen.getByText(/Running Queue/i)).toBeInTheDocument();
  });

  it('renders ComparisonWorkspaceView and toggles between Experiments and Mechanisms', () => {
    render(<ComparisonWorkspaceView />);

    expect(screen.getByText(/Scientific Comparison Workspace/i)).toBeInTheDocument();
    expect(screen.getByText(/Experiment A: L9H9 Zero-Ablation/i)).toBeInTheDocument();
    expect(screen.getByText(/Experiment B: L0H0 Control Ablation/i)).toBeInTheDocument();

    const mechTabBtn = screen.getByText(/Mechanism Comparison/i);
    fireEvent.click(mechTabBtn);

    expect(screen.getByText(/Mechanism Alpha: L8H1 ➔ L9H9 ➔ Residual Stream/i)).toBeInTheDocument();
    expect(screen.getByText(/Mechanism Beta: L8H1 ➔ MLP_L8 ➔ Residual Stream/i)).toBeInTheDocument();
  });

  it('renders WelcomeOnboardingModal with distinct launch pathways', () => {
    const handleSelectAction = vi.fn();
    const handleClose = vi.fn();

    render(
      <WelcomeOnboardingModal
        isOpen={true}
        onClose={handleClose}
        onSelectAction={handleSelectAction}
      />
    );

    expect(screen.getByText(/Welcome to MECH Research OS/i)).toBeInTheDocument();
    expect(screen.getByText(/Start a New Investigation/i)).toBeInTheDocument();
    expect(screen.getByText(/Explore a Known Literature Mechanism/i)).toBeInTheDocument();
    expect(screen.getByText(/Open Research Package \(\.mech bundle\)/i)).toBeInTheDocument();

    fireEvent.click(screen.getByText(/Explore a Known Literature Mechanism/i));
    expect(handleSelectAction).toHaveBeenCalledWith('EXPLORE_BENCHMARKS');
  });

  it('renders UniversalInspector with Why-derivation chain and provenance tags', () => {
    const handleOpenMethodology = vi.fn();

    render(<UniversalInspector onOpenMethodology={handleOpenMethodology} />);

    expect(screen.getByText(/Universal Scientific Inspector/i)).toBeInTheDocument();
    expect(screen.getByText(/Why am I seeing this\?/i)).toBeInTheDocument();
    expect(screen.getByText(/Measured Metrics & Provenance/i)).toBeInTheDocument();
    expect(screen.getByText(/Supporting Empirical Evidence/i)).toBeInTheDocument();
    expect(screen.getByText(/Scientific Limitations/i)).toBeInTheDocument();

    const methodBtn = screen.getByText(/View Exact Experimental Methodology/i);
    fireEvent.click(methodBtn);
    expect(handleOpenMethodology).toHaveBeenCalled();
  });
});
