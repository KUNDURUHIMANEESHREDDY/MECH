import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { BenchmarkPanel } from '../../src/components/panels/BenchmarkPanel';

describe('VisualizationSprint4', () => {
  it('renders BenchmarkPanel with benchmark suites', () => {
    render(<BenchmarkPanel />);
    expect(screen.getAllByText(/Benchmark/i)[0]).toBeInTheDocument();
    expect(screen.getAllByText(/Cases/i)[0]).toBeInTheDocument();

  });
});
