import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import ExperimentNotebook from '../../src/components/ExperimentNotebook.jsx';
import SessionTimelineView from '../../src/components/SessionTimelineView.jsx';
import { WorkspaceBundle } from '../../src/domain/workspace/workspaceExporter.js';
import { PluginLoader } from '../../src/domain/plugins/pluginLoader.js';
import { ReportGenerator } from '../../src/domain/reports/reportGenerator.js';

describe('Sprint 2 Deliverable UI & Round-Trip Tests', () => {
  it('renders ExperimentNotebook and SessionTimelineView', () => {
    render(<ExperimentNotebook />);
    render(<SessionTimelineView />);
    expect(screen.getByTestId('experiment-notebook')).toBeInTheDocument();
    expect(screen.getByTestId('session-timeline-view')).toBeInTheDocument();
  });

  it('performs workspace export and import round-trip', () => {
    const bundle = WorkspaceBundle.exportBundle();
    expect(bundle.version).toBe('2.0.0');
    expect(bundle.notebook).toBeDefined();

    const importRes = WorkspaceBundle.importBundle(bundle);
    expect(importRes.success).toBe(true);
  });

  it('validates plugin manifest loading', () => {
    const manifest = {
      id: 'plugin_sae_explorer',
      name: 'SAE Deep Explorer',
      version: '1.0.0',
      minimumRuntimeVersion: '2.0.0',
    };
    const loaded = PluginLoader.loadPlugin(manifest);
    expect(loaded.status).toBe('active');
  });

  it('generates markdown report', () => {
    const md = ReportGenerator.generateMarkdownReport('Sprint 2 Final Report');
    expect(md).toContain('Sprint 2 Final Report');
    expect(md).toContain('GPT-2 Small');
  });
});
