const { test, expect } = require('@playwright/test');

// Canonical route labels. Hash IDs remain compatible with deep links.
const NAV_PAGES = [
  ['explorer', 'Model Explorer'],
  ['gpt2', 'GPT-2 Live'],
  ['transformer', 'Transformer Visualizer'],
  ['workspace', 'Workspace'],
  ['neuralexplorer', 'Neural Explorer'],
  ['knowledgegraph', 'Knowledge Graph'],
  ['circuitexplorer', 'Circuit Explorer'],
  ['models', 'Models'],
  ['prompts', 'Prompts'],
  ['debugger', 'Debugger'],
  ['build', 'Build'],
  ['benchmark', 'Benchmark'],
  ['benchmarksuite', 'Benchmark Suite'],
  ['experiments', 'Experiments'],
  ['sessions', 'Sessions'],
  ['reports', 'Reports'],
  ['reasoning', 'Reasoning'],
  ['evidencefusion', 'Evidence Fusion'],
  ['analytics', 'Analytics'],
  ['health', 'Health'],
  ['campaigns', 'Campaigns'],
  ['plugins', 'Plugins'],
  ['notebook', 'Research Notebook'],
  ['labnotebook', 'Lab Notebook'],
  ['reproduction', 'Paper Reproduction'],
  ['settings', 'Settings'],
  ['logging', 'Logging'],
  ['projects', 'Projects'],
  ['recent', 'Recent Files'],
];

const HASH_ONLY_PAGES = [
  ['gpt2explorer', 'GPT-2 Neuron Explorer'],
  ['transformerExplorer', 'Transformer Explorer'],
];

test('sidebar exposes every navigation item', async ({ page }) => {
  await page.goto('/');
  for (const [key] of NAV_PAGES) {
    await expect(page.getByTestId(`nav-${key}`)).toBeVisible();
  }
});

test('sidebar navigation shows one titled tool view and updates the hash', async ({ page }) => {
  await page.goto('/');
  for (const [key] of NAV_PAGES) {
    const item = page.getByTestId(`nav-${key}`);
    await item.scrollIntoViewIfNeeded();
    await item.click();
    await expect(page).toHaveURL(new RegExp(`#${key}$`));
    const view = page.getByTestId(`view-${key}`);
    await expect(view).toBeVisible();
    await expect(view.getByRole('heading').first()).toBeVisible();
  }
});

test('hash-only pages open without a model gate', async ({ page }) => {
  for (const [key] of HASH_ONLY_PAGES) {
    await page.goto(`/#${key}`);
    const view = page.getByTestId(`view-${key}`);
    await expect(view).toBeVisible();
    await expect(view.getByRole('heading').first()).toBeVisible();
  }
});

test('navigating away replaces the visible tool instead of stacking', async ({ page }) => {
  await page.goto('/#explorer');
  await expect(page.getByTestId('view-explorer')).toBeVisible();

  await page.getByTestId('nav-society').click();
  await expect(page.getByTestId('view-society')).toBeVisible();
  await expect(page.getByTestId('view-explorer')).toHaveCount(0);
});

test('sidebar status exposes Python/runtime state', async ({ page }) => {
  await page.goto('/');
  const status = page.getByTestId('runtime-status');
  await expect(status).toBeVisible();
  await expect(status).toContainText(/Connecting|Connected|Offline|models available|No model selected/);
});
