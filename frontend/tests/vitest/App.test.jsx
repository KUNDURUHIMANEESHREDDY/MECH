import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import App from '../../src/App.tsx';
import Sidebar from '../../src/components/Sidebar.jsx';
import ThemeSettings from '../../src/pages/ThemeSettings.jsx';

describe('App shell', () => {
  beforeEach(() => {
    document.documentElement.removeAttribute('data-theme');
  });

  it('renders the sidebar with Neural Debugger navigation items', async () => {
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

  it('navigates to settings when the sidebar item is clicked', async () => {
    render(<App />);
    fireEvent.click(screen.getByTestId('nav-settings'));
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'Settings' })).toBeInTheDocument();
    });
  });
});

describe('Sidebar', () => {
  const pages = {
    workspace: { label: 'Workspace', component: () => null },
    models: { label: 'Models', component: () => null }
  };

  it('marks the active item', () => {
    render(<Sidebar pages={pages} active="models" onSelect={() => {}} />);
    const item = screen.getByTestId('nav-models');
    expect(item.className).toMatch(/active/);
  });

  it('fires onSelect on click and keyboard activation', () => {
    const onSelect = vi.fn();
    render(<Sidebar pages={pages} active="workspace" onSelect={onSelect} />);
    fireEvent.click(screen.getByTestId('nav-models'));
    expect(onSelect).toHaveBeenCalledWith('models');
    fireEvent.keyDown(screen.getByTestId('nav-models'), { key: 'Enter' });
    expect(onSelect).toHaveBeenCalledTimes(2);
  });
});

describe('ThemeSettings', () => {
  it('renders the current theme selection', () => {
    render(<ThemeSettings settings={{ theme: 'dark' }} onChange={() => {}} />);
    const select = screen.getByTestId('theme-select');
    expect(select.value).toBe('dark');
  });

  it('invokes onChange with the new theme', () => {
    const onChange = vi.fn();
    render(<ThemeSettings settings={{ theme: 'system' }} onChange={onChange} />);
    fireEvent.change(screen.getByTestId('theme-select'), { target: { value: 'light' } });
    expect(onChange).toHaveBeenCalledWith({ theme: 'light' });
  });
});
