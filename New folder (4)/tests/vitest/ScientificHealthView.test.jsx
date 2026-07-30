import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import { describe, it, expect } from 'vitest';
import ScientificHealthView from '../../src/components/ScientificHealthView.jsx';

describe('ScientificHealthView Component', () => {
  it('renders Scientific Health header and health score cards', () => {
    render(<ScientificHealthView />);
    expect(screen.getByText(/Scientific Health & Continuous Validation/i)).toBeInTheDocument();
    expect(screen.getByText(/Platform Health Score/i)).toBeInTheDocument();
    expect(screen.getByText(/Reproducibility Score/i)).toBeInTheDocument();
    expect(screen.getByText(/Benchmark Pass Rate/i)).toBeInTheDocument();
  });

  it('renders golden benchmarks validation matrix table', () => {
    render(<ScientificHealthView />);
    expect(screen.getByText(/Golden Benchmarks Validation Matrix/i)).toBeInTheDocument();
    expect(screen.getAllByText(/IOI Circuit Recovery/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Induction Head Sequence Repeater/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/SAE Feature Dictionary Recovery/i).length).toBeGreaterThan(0);
  });

  it('renders actionable scientific alerts feed', () => {
    render(<ScientificHealthView />);
    expect(screen.getByText(/Actionable Scientific Alerts/i)).toBeInTheDocument();
    expect(screen.getByText(/Platform Health Optimal/i)).toBeInTheDocument();
  });

  it('renders runtime environment profile', () => {
    render(<ScientificHealthView />);
    expect(screen.getByText(/Runtime Environment Profile/i)).toBeInTheDocument();
    expect(screen.getByText(/PyTorch Version/i)).toBeInTheDocument();
    expect(screen.getByText(/Transformers Version/i)).toBeInTheDocument();
  });

  it('renders Run Validation Suite button and handles click', () => {
    render(<ScientificHealthView />);
    const btn = screen.getByText(/Run Validation Suite/i);
    expect(btn).toBeInTheDocument();
    fireEvent.click(btn);
    expect(screen.getByText(/Running Validation/i)).toBeInTheDocument();
  });
});
