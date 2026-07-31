import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import { describe, it, expect } from 'vitest';
import KnowledgeGraphView from '../../src/components/KnowledgeGraphView.jsx';

describe('KnowledgeGraphView Component', () => {
  it('renders knowledge graph title and perspective tabs', () => {
    render(<KnowledgeGraphView />);
    expect(screen.getByText(/Scientific Knowledge Graph/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Mechanism View/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Paper View/i)).toBeInTheDocument();
  });

  it('switches exploratory perspective when tab button is clicked', () => {
    render(<KnowledgeGraphView />);
    const paperTab = screen.getByText(/Paper View/i);
    fireEvent.click(paperTab);
    expect(screen.getByText(/Trace academic paper citations down to claims/i)).toBeInTheDocument();
    expect(screen.getByText(/Wang et al. \(2022\) IOI Paper/i)).toBeInTheDocument();
  });

  it('renders graph query console buttons', () => {
    render(<KnowledgeGraphView />);
    expect(screen.getByText(/Show every experiment supporting IOI Name Mover/i)).toBeInTheDocument();
    expect(screen.getByText(/Which papers discuss GPT-2 L9H9\?/i)).toBeInTheDocument();
  });
});
