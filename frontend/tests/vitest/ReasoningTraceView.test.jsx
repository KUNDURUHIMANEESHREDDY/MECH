import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ReasoningTraceView } from '../../src/components/ReasoningTraceView';

describe('ReasoningTraceView', () => {
  it('renders reasoning timeline title and filters', () => {
    render(<ReasoningTraceView />);
    expect(screen.getByText(/Reasoning Trace/i)).toBeInTheDocument();
  });
});
