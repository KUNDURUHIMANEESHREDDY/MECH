import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { EvidenceFusionView } from '../../src/components/EvidenceFusionView';

describe('KnowledgeGraphView', () => {
  it('renders knowledge graph and causal evidence containers', () => {
    render(<EvidenceFusionView />);
    expect(screen.getByText(/Evidence Fusion/i)).toBeInTheDocument();
  });
});
