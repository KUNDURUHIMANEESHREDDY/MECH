import { describe, expect, it } from 'vitest';
import { notebookStore } from '../../src/domain/notebook/notebookStore';
import { ReportGenerator } from '../../src/domain/reports/reportGenerator';
import { WorkspaceBundle } from '../../src/domain/workspace/workspaceExporter';

const BANNED = [
  'L8_N402',
  'L8_N402 (IOI)',
  '0.82',
  'firing_freq',
  'sae_gpt2_l8.pt',
  'Replacement patch applied',
  'Prediction Delta',
];

function seedSnapshot() {
  return JSON.stringify(new (notebookStore.constructor)().getCells());
}

describe('frontend fabrication guards', () => {
  it('fresh notebook seeds carry no numeric findings', () => {
    const text = seedSnapshot();
    for (const banned of BANNED) {
      expect(text).not.toContain(banned);
    }
    const cells = JSON.parse(text);
    const hasNumbers = cells.some((c) => JSON.stringify(c).match(/\d+\.\d+/));
    expect(hasNumbers).toBe(false);
  });

  it('generated reports assert no hardcoded interventions', () => {
    const md = ReportGenerator.generateMarkdownReport('Probe');
    for (const banned of BANNED) {
      expect(md).not.toContain(banned);
    }
    expect(md).toMatch(/no measurements recorded/i);
    expect(md).toMatch(/provenance/i);
  });

  it('workspace export metadata is unconfirmed, never asserted', () => {
    const bundle = WorkspaceBundle.exportBundle();
    expect(bundle.metadata.active_model).not.toBe('GPT-2 Small');
    expect(bundle.metadata.dataset).not.toBe('IOI Benchmark');
    expect(bundle.metadata.active_model).toBe('unconfirmed');
    expect(bundle.metadata.dataset).toBe('unconfirmed');
  });

  it('dead fake-data services stay deleted', async () => {
    // @vite-ignore keeps these runtime-checked: a static import would fail
    // the whole suite at transform time instead of asserting absence.
    const svc = '../../src/services/' + 'visualizationService.js';
    const adapter = '../../src/utils/' + 'visualizationAdapter.js';
    await expect(import(/* @vite-ignore */ svc)).rejects.toThrow();
    await expect(import(/* @vite-ignore */ adapter)).rejects.toThrow();
  });
});
