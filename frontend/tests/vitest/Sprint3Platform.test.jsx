import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ExperimentsView } from '../../src/components/ExperimentsView';
import { ExperimentNotebook } from '../../src/components/ExperimentNotebook';

describe('Sprint3Platform', () => {
  it('renders ExperimentsView with active experiment runners', () => {
    render(<ExperimentsView />);
    expect(screen.getByText('Experiments')).toBeInTheDocument();
  });

  it('renders ExperimentNotebook component', () => {
    render(<ExperimentNotebook />);
    expect(screen.getByText('Experiment Notebook')).toBeInTheDocument();
  });
});
