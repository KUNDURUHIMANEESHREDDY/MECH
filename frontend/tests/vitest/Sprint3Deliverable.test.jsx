import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { Projects } from '../../src/components/Projects';
import { SessionsView } from '../../src/components/SessionsView';
import { PromptsView } from '../../src/components/PromptsView';

describe('Sprint3Deliverable', () => {
  it('renders Projects component with title and creation fields', () => {
    render(<Projects />);
    expect(screen.getByText(/Research Projects/i)).toBeInTheDocument();
  });

  it('renders SessionsView component with session controls', () => {
    render(<SessionsView />);
    expect(screen.getByText(/Saved Sessions/i)).toBeInTheDocument();
  });

  it('renders PromptsView component with prompt management', () => {
    render(<PromptsView />);
    expect(screen.getByText(/Prompts Library/i)).toBeInTheDocument();
  });
});
