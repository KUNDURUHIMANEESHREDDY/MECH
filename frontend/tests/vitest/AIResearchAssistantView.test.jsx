import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { AIResearchAssistantView } from '../../src/components/AIResearchAssistantView';

describe('AIResearchAssistantView', () => {
  it('renders AI Research Assistant title and goal formulation card', () => {
    render(<AIResearchAssistantView />);
    expect(screen.getByText(/AI Research Assistant/i)).toBeInTheDocument();
    expect(screen.getByText(/Research Goal/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Run Investigation/i })).toBeInTheDocument();
  });

  it('renders preset investigation chips', () => {
    render(<AIResearchAssistantView />);
    expect(screen.getByText(/Fact Recall/i)).toBeInTheDocument();
    expect(screen.getByText(/Indirect Object Identification/i)).toBeInTheDocument();
    expect(screen.getByText(/Greater-Than Comparison Logic/i)).toBeInTheDocument();
    expect(screen.getByText(/Hallucination Circuit Localization/i)).toBeInTheDocument();
  });
});

