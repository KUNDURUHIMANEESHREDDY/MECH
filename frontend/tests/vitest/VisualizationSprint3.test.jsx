import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { KnowledgeGraphPanel } from '../../src/components/panels/KnowledgeGraphPanel';

describe('VisualizationSprint3', () => {
  it('renders KnowledgeGraphPanel with graph topology', () => {
    render(<KnowledgeGraphPanel />);
    expect(screen.getByText(/Evidence-Aware Mechanistic Knowledge Graph/i)).toBeInTheDocument();
  });
});
