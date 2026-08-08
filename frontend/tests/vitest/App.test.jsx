import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import App from '../../src/App.tsx';
import { Navigator } from '../../src/shell/navigator/Navigator';

describe('App shell', () => {
  beforeEach(() => {
    document.documentElement.removeAttribute('data-theme');
  });

  it('renders the navigator with navigation items', async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByTestId('nav-workspace')).toBeInTheDocument();
      expect(screen.getByTestId('nav-models')).toBeInTheDocument();
      expect(screen.getByTestId('nav-prompts')).toBeInTheDocument();
      expect(screen.getByTestId('nav-debugger')).toBeInTheDocument();
      expect(screen.getByTestId('nav-experiments')).toBeInTheDocument();
      expect(screen.getByTestId('nav-sessions')).toBeInTheDocument();
      expect(screen.getByTestId('nav-reports')).toBeInTheDocument();
      expect(screen.getByTestId('nav-settings')).toBeInTheDocument();
      expect(screen.getByTestId('nav-logging')).toBeInTheDocument();
      expect(screen.getByTestId('nav-build')).toBeInTheDocument();
    });
  });

  it('shows the python status indicator in the topbar', async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByTestId('python-status')).toBeInTheDocument();
    });
  });

  it('updates the breadcrumb when a nav item is clicked', async () => {
    render(<App />);
    fireEvent.click(screen.getByTestId('nav-settings'));
    await waitFor(() => {
      expect(screen.getByTestId('topbar-crumb')).toHaveTextContent('Settings');
    });
  });
});

describe('Navigator', () => {
  it('renders resource tree groups', () => {
    render(<Navigator />);
    expect(screen.getByText('Models & Architectures')).toBeInTheDocument();
    expect(screen.getByText('Mechanistic Analysis')).toBeInTheDocument();
    expect(screen.getByText('Workspace & Assets')).toBeInTheDocument();
  });

  it('expands groups on click', () => {
    render(<Navigator />);
    // Groups start expanded by default, so GPT-2 is visible
    expect(screen.getByText('GPT-2 Live')).toBeInTheDocument();
    // Click to collapse the group
    const groups = screen.getAllByText('Models & Architectures');
    fireEvent.click(groups[0]);
    expect(screen.queryByText('GPT-2 Live')).not.toBeInTheDocument();
    // Click again to expand
    fireEvent.click(groups[0]);
    expect(screen.getByText('GPT-2 Live')).toBeInTheDocument();
  });
});
