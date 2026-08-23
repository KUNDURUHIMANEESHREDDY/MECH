import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { BuildLog } from '../../src/components/BuildLog';
import { Logging } from '../../src/components/Logging';
import { StatusBar } from '../../src/components/StatusBar';

describe('Sprint4Deliverable', () => {
  it('renders BuildLog component with build action controls', () => {
    render(<BuildLog />);
    expect(screen.getByText(/Build Log/i)).toBeInTheDocument();
  });

  it('renders Logging component with log levels', () => {
    render(<Logging />);
    expect(screen.getByText(/Execution Logs/i)).toBeInTheDocument();
  });

  it('renders StatusBar component', () => {
    render(<StatusBar />);
    expect(screen.getByTestId('python-status')).toBeInTheDocument();
  });
});
