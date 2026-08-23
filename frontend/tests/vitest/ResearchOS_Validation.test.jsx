import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { MethodologicalLimitationsBadge } from '../../src/components/research/MethodologicalLimitationsBadge';
import { ClaimDerivationModal } from '../../src/components/research/ClaimDerivationModal';
import { ResearchTimelineView } from '../../src/components/research/ResearchTimelineView';
import { EpistemicStateView } from '../../src/components/research/EpistemicStateView';
import { AlternativeMechanismComparison } from '../../src/components/research/AlternativeMechanismComparison';

describe('MECH Research Validation v1 - Scientific UX Components', () => {
  it('renders MethodologicalLimitationsBadge and toggles details', () => {
    render(<MethodologicalLimitationsBadge method="ATTENTION" />);
    expect(screen.getByText(/Attention Weight Caveat/i)).toBeInTheDocument();
    
    // Toggle details
    fireEvent.click(screen.getByText(/Attention Weight Caveat/i));
    expect(screen.getByText(/High attention weight does not imply/i)).toBeInTheDocument();
  });

  it('renders ClaimDerivationModal with 6-stage derivation pipeline', () => {
    const handleClose = vi.fn();
    render(
      <ClaimDerivationModal
        claimTitle="L9H9 mediates indirect-object token routing"
        targetComponent="L9H9"
        onClose={handleClose}
      />
    );
    expect(screen.getByText(/Why is this claim valid\?/i)).toBeInTheDocument();
    expect(screen.getByText(/Attention Routing/i)).toBeInTheDocument();
    expect(screen.getByText(/Negative Control \(L0H0\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Alternative A: L8H4 Backup Head/i)).toBeInTheDocument();
  });

  it('renders ResearchTimelineView with chronological event stream and filters', () => {
    render(<ResearchTimelineView />);
    expect(screen.getByText(/Research Action Timeline/i)).toBeInTheDocument();
    expect(screen.getByText(/Investigation Initialized/i)).toBeInTheDocument();
    expect(screen.getByText(/Causal Ablation Executed/i)).toBeInTheDocument();
    expect(screen.getByText(/Falsification Test Passed/i)).toBeInTheDocument();
  });

  it('renders EpistemicStateView with 5 epistemic categories', () => {
    render(<EpistemicStateView />);
    expect(screen.getByText(/What Do We Actually Know\?/i)).toBeInTheDocument();
    expect(screen.getByText(/Established Facts/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Causally Supported/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Candidate Hypotheses/i)).toBeInTheDocument();
    expect(screen.getByText(/Unknowns & Unexamined/i)).toBeInTheDocument();
    expect(screen.getByText(/Contradicted \/ Falsified/i)).toBeInTheDocument();
  });

  it('renders AlternativeMechanismComparison with side-by-side empirical metrics', () => {
    render(<AlternativeMechanismComparison />);
    expect(screen.getByText(/Alternative Mechanism Comparison/i)).toBeInTheDocument();
    expect(screen.getByText(/Primary Hypothesis \(H1\): L9H9 Name Mover/i)).toBeInTheDocument();
    expect(screen.getByText(/Alternative A: L8H4 Backup Head/i)).toBeInTheDocument();
    expect(screen.getByText(/Alternative B: MLP Layer 8 Direct Computation/i)).toBeInTheDocument();
  });
});
