import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { collaborationEngine } from '../../src/services/collaborationEngine';
import { visualizationEngine } from '../../src/services/visualizationEngine';

describe('Sprint 4 End-to-End Deliverable Integration', () => {
  it('verifies collaborationEngine workspace overview', () => {
    const overview = collaborationEngine.getWorkspaceOverview();
    expect(overview.users.length).toBeGreaterThanOrEqual(3);
    expect(overview.experiments.length).toBeGreaterThanOrEqual(2);
    expect(overview.datasets.length).toBeGreaterThanOrEqual(2);
  });

  it('verifies arXiv publication pipeline compilation', () => {
    const pkg = collaborationEngine.publicationPipeline.compilePackage('# Title', 'Autonomous Interp Paper');
    expect(pkg.status).toBe('ReadyForSubmission');
    expect(pkg.figuresCount).toBe(4);
  });

  it('verifies visualizationEngine view export', () => {
    const exp = visualizationEngine.exportCurrentView('SVG');
    expect(exp.format).toBe('SVG');
  });
});
