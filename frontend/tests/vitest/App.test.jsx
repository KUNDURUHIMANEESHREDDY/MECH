import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import App from '../../src/App.tsx';
import { FeaturesDrawer } from '../../src/shell/features/FeaturesDrawer';
import { useUIStore } from '../../src/shared/stores/ui';

describe('App shell', () => {
  beforeEach(() => {
    document.documentElement.removeAttribute('data-theme');
    useUIStore.setState({ sidebarCollapsed: false });
  });

  it('renders the navigator with navigation items', async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByTestId('nav-active_investigation')).toBeInTheDocument();
      expect(screen.getByTestId('nav-models')).toBeInTheDocument();
      expect(screen.getByTestId('nav-model_explorer')).toBeInTheDocument();
      expect(screen.getByTestId('nav-settings')).toBeInTheDocument();
      expect(screen.getByTestId('nav-logging')).toBeInTheDocument();
      expect(screen.getByTestId('nav-compute_center')).toBeInTheDocument();
      expect(screen.getByTestId('nav-attention_heatmap')).toBeInTheDocument();
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

describe('FeaturesDrawer', () => {
  beforeEach(() => {
    useUIStore.setState({ sidebarCollapsed: false });
  });

  it('renders resource tree groups', () => {
    render(<FeaturesDrawer />);
    expect(screen.getByText('Workspace')).toBeInTheDocument();
    expect(screen.getByText('Model')).toBeInTheDocument();
    expect(screen.getByText('Analysis')).toBeInTheDocument();
    expect(screen.getByText('Research')).toBeInTheDocument();
    expect(screen.getByText('System')).toBeInTheDocument();
  });

  it('expands groups on click', () => {
    render(<FeaturesDrawer />);
    // Groups start expanded by default, so nav items are visible
    expect(screen.getByText('Overview (Active Investigation)')).toBeInTheDocument();
    // Click to collapse the Workspace group
    const groups = screen.getAllByText('Workspace');
    fireEvent.click(groups[0]);
    expect(screen.queryByText('Overview (Active Investigation)')).not.toBeInTheDocument();
    // Click again to expand
    fireEvent.click(groups[0]);
    expect(screen.getByText('Overview (Active Investigation)')).toBeInTheDocument();
  });
});
