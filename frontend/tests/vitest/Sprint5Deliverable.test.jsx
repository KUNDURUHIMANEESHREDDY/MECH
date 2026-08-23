import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ReportsView } from '../../src/components/ReportsView';
import { ResearchNotebook } from '../../src/components/ResearchNotebook';

describe('Sprint5Deliverable', () => {
  it('renders ReportsView component with report generator', () => {
    render(<ReportsView />);
    expect(screen.getByText(/Research Reports/i)).toBeInTheDocument();
  });

  it('renders ResearchNotebook component', () => {
    render(<ResearchNotebook />);
    expect(screen.getByText(/Research Notebook/i)).toBeInTheDocument();
  });
});
