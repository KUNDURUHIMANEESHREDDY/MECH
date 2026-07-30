import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import ReasoningTraceView from '../../src/components/ReasoningTraceView.jsx';

describe('ReasoningTraceView Component', () => {
  it('renders scientific reasoning header and observation', () => {
    render(<ReasoningTraceView />);
    expect(screen.getByText(/Scientific Reasoning/i)).toBeInTheDocument();
    expect(screen.getByText(/Feature_1042/i)).toBeInTheDocument();
  });

  it('renders trace type selector tabs', () => {
    render(<ReasoningTraceView />);
    expect(screen.getAllByText(/Debate/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Discovery/i).length).toBeGreaterThan(0);
  });

  it('switches trace type when tab is clicked', () => {
    render(<ReasoningTraceView />);
    const discoveryTab = screen.getAllByText(/Discovery/i)[0];
    fireEvent.click(discoveryTab);
    expect(screen.getByText(/ACDC Circuit #7/i)).toBeInTheDocument();
  });

  it('filters hypotheses via search input', () => {
    render(<ReasoningTraceView />);
    const searchInput = screen.getByPlaceholderText(/Search traces/i);
    fireEvent.change(searchInput, { target: { value: 'French' } });
    expect(screen.getByText(/Fires on French cities/i)).toBeInTheDocument();
  });

  it('triggers navigation callback when clickable node is clicked', () => {
    const onNavigate = vi.fn();
    render(<ReasoningTraceView onNavigate={onNavigate} />);
    const link = screen.getAllByText(/Neural Explorer/i)[0];
    fireEvent.click(link);
    expect(onNavigate).toHaveBeenCalledWith('neuralexplorer');
  });
});
