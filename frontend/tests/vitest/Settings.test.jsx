import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import Settings from '../../src/components/Settings.jsx';
import Projects from '../../src/components/Projects.jsx';

const api = window.appApi;

describe('Settings page', () => {
  it('renders all settings sections', () => {
    const settings = {
      theme: 'dark',
      gpu: { enabled: true, acceleration: 'auto' },
      cache: { enabled: true, maxSizeMb: 256, location: '' },
      paths: { python: 'python', workspace: '', projects: '' }
    };
    render(<Settings settings={settings} onChange={() => {}} api={api} />);
    expect(screen.getByRole('heading', { name: 'Settings' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Theme' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'GPU' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Cache' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Paths' })).toBeInTheDocument();
  });

  it('shows a loading state when settings are not yet loaded', () => {
    render(<Settings settings={null} onChange={() => {}} api={api} />);
    expect(screen.getByText(/Loading settings/)).toBeInTheDocument();
  });
});

describe('Projects page', () => {
  it('renders an empty list when there are no projects', async () => {
    render(<Projects api={api} />);
    await waitFor(() => {
      expect(screen.getByText('No projects yet.')).toBeInTheDocument();
    });
  });
});
