import React from 'react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { useResearchStore } from '../../src/shared/stores/research';
import { ActiveInvestigationOverview } from '../../src/components/research/ActiveInvestigationOverview';
import { HypothesisLab } from '../../src/components/research/HypothesisLab';
import { InterventionLab } from '../../src/components/research/InterventionLab';
import { EvidenceGraphView } from '../../src/components/research/EvidenceGraphView';
import { MechanismBuilder } from '../../src/components/research/MechanismBuilder';

// Mock api service
vi.mock('../../src/services/api', () => ({
  api: {
    listInvestigations: vi.fn().mockResolvedValue({
      investigations: [
        {
          id: 'inv_ioi_test',
          title: 'Indirect Object Identification Circuit',
          research_question: 'How does GPT-2 retrieve the indirect object?',
          model_id: 'gpt2',
          dataset_id: 'ioi',
          status: 'ACTIVE',
          tags: ['ioi', 'circuits'],
        },
      ],
    }),
    getInvestigation: vi.fn().mockResolvedValue({
      investigation: {
        id: 'inv_ioi_test',
        title: 'Indirect Object Identification Circuit',
        research_question: 'How does GPT-2 retrieve the indirect object?',
      },
      hypotheses: [
        {
          id: 'hyp_l9h9',
          investigation_id: 'inv_ioi_test',
          title: 'L9H9 Name Mover',
          statement: 'L9H9 moves indirect object token.',
          target_component: 'L9H9',
          status: 'SUPPORTED',
          evidence_count_supporting: 2,
          evidence_count_contradicting: 0,
        },
      ],
      runs: [],
      evidence: [],
      mechanisms: [],
    }),
    listHypotheses: vi.fn().mockResolvedValue({ hypotheses: [] }),
    getEvidenceMatrix: vi.fn().mockResolvedValue({ matrix: [] }),
    listMechanisms: vi.fn().mockResolvedValue({ mechanisms: [] }),
    listArtifacts: vi.fn().mockResolvedValue({ artifacts: [] }),
    generateResearchReport: vi.fn().mockResolvedValue({ report_markdown: '# Report' }),
    listJobs: vi.fn().mockResolvedValue({ jobs: [] }),
  },
}));

describe('Research OS Frontend Components & Store', () => {
  beforeEach(() => {
    useResearchStore.setState({
      investigations: [
        {
          id: 'inv_ioi_test',
          title: 'Indirect Object Identification Circuit',
          research_question: 'How does GPT-2 retrieve the indirect object?',
          model_id: 'gpt2',
          dataset_id: 'ioi',
          status: 'ACTIVE',
          tags: ['ioi', 'circuits'],
          created_at: Date.now(),
          updated_at: Date.now(),
        },
      ],
      activeInvestigationId: 'inv_ioi_test',
      activeInvestigation: {
        id: 'inv_ioi_test',
        title: 'Indirect Object Identification Circuit',
        research_question: 'How does GPT-2 retrieve the indirect object?',
        model_id: 'gpt2',
        dataset_id: 'ioi',
        status: 'ACTIVE',
        tags: ['ioi', 'circuits'],
        created_at: Date.now(),
        updated_at: Date.now(),
      },
      hypotheses: [
        {
          id: 'hyp_l9h9',
          investigation_id: 'inv_ioi_test',
          title: 'L9H9 Name Mover',
          statement: 'L9H9 moves indirect object token.',
          target_component: 'L9H9',
          prediction: 'Ablation drops target prob.',
          expected_evidence: 'Δlogit > 1.0',
          falsification_condition: 'Δlogit < 0.2',
          status: 'SUPPORTED',
          evidence_count_supporting: 2,
          evidence_count_contradicting: 0,
          created_at: Date.now(),
          updated_at: Date.now(),
        },
      ],
      activeHypothesisId: 'hyp_l9h9',
      activeHypothesis: {
        id: 'hyp_l9h9',
        investigation_id: 'inv_ioi_test',
        title: 'L9H9 Name Mover',
        statement: 'L9H9 moves indirect object token.',
        target_component: 'L9H9',
        prediction: 'Ablation drops target prob.',
        expected_evidence: 'Δlogit > 1.0',
        falsification_condition: 'Δlogit < 0.2',
        status: 'SUPPORTED',
        evidence_count_supporting: 2,
        evidence_count_contradicting: 0,
        created_at: Date.now(),
        updated_at: Date.now(),
      },
      runs: [],
      evidence: [],
      evidenceMatrix: [],
      mechanisms: [],
      selectedComponent: { name: 'L9H9', layer: 9, head: 9, componentType: 'head' },
      selectedTokenIndex: null,
      isDemoMode: false,
    });
  });

  it('renders ActiveInvestigationOverview with scientific question and active hypothesis', () => {
    render(<ActiveInvestigationOverview />);
    expect(screen.getByText(/Active Scientific Investigation/i)).toBeDefined();
    expect(screen.getByText(/Indirect Object Identification Circuit/i)).toBeDefined();
    expect(screen.getByText(/L9H9 Name Mover/i)).toBeDefined();
  });

  it('renders HypothesisLab and lists active hypotheses', () => {
    render(<HypothesisLab />);
    expect(screen.getByText(/Hypothesis Formation & Falsification Engine/i)).toBeDefined();
    expect(screen.getByText(/L9H9 Name Mover/i)).toBeDefined();
  });

  it('renders InterventionLab with causal execution controls', () => {
    render(<InterventionLab />);
    expect(screen.getByText(/Causal Intervention & Patching Lab/i)).toBeDefined();
    expect(screen.getByText(/Execute Causal Intervention/i)).toBeDefined();
  });

  it('renders EvidenceGraphView with scientific provenance nodes', () => {
    render(<EvidenceGraphView />);
    expect(screen.getByText(/Evidence & Knowledge Architecture/i)).toBeDefined();
  });

  it('renders MechanismBuilder with visual pipeline stages', () => {
    render(<MechanismBuilder />);
    expect(screen.getByText(/Visual Mechanism Builder/i)).toBeDefined();
    expect(screen.getByText(/EVIDENCE-BACKED/i)).toBeDefined();
  });

  it('renders InvestigationHeader with active question and presets', async () => {
    const { InvestigationHeader } = await import('../../src/components/research/InvestigationHeader');
    render(<InvestigationHeader />);
    expect(screen.getByText(/INVESTIGATION/i)).toBeDefined();
    expect(screen.getByText(/Indirect Object Identification Circuit/i)).toBeDefined();
    expect(screen.getByText(/L9H9 Name Mover/i)).toBeDefined();
  });

  it('renders Inspector with 6 object tabs', async () => {
    const { Inspector } = await import('../../src/shell/inspector/Inspector');
    render(<Inspector />);
    expect(screen.getByText(/Universal Object Inspector/i)).toBeDefined();
    expect(screen.getByText(/Overview/i)).toBeDefined();
    expect(screen.getByText(/Identity/i)).toBeDefined();
  });

  it('renders ResearchIntegrityPanel with scientific integrity gates', async () => {
    const { ResearchIntegrityPanel } = await import('../../src/components/research/ResearchIntegrityPanel');
    render(<ResearchIntegrityPanel />);
    expect(screen.getByText(/Research Integrity & Production Gate v1/i)).toBeDefined();
    expect(screen.getByText(/Model Weight Provenance/i)).toBeDefined();
    expect(screen.getByText(/Two-Phase Atomic Tensor Storage/i)).toBeDefined();
    expect(screen.getByText(/Export Research Bundle/i)).toBeDefined();
  });
});
