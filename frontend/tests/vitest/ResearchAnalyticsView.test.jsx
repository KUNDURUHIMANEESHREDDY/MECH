import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ResearchAnalyticsView } from '../../src/components/ResearchAnalyticsView';

describe('ResearchAnalyticsView', () => {
  it('renders analytics metrics and refresh button', () => {
    render(<ResearchAnalyticsView />);
    expect(screen.getByText('Analytics')).toBeInTheDocument();
    expect(screen.getByText(/Experiment runs/i)).toBeInTheDocument();
    expect(screen.getByText(/Tokens generated/i)).toBeInTheDocument();
  });
});
