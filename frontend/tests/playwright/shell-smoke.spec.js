const { test, expect } = require('@playwright/test');

// Labels must match App.tsx PAGES (source of the Topbar breadcrumb).
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

// The renderer runs standalone in these e2e tests (no Python backend at
// localhost:8000), so listModels() rejects and ErrorUI overlays the app.
// The rejection is async relative to page.goto (and the dev-mode StrictMode
// double-mount fires listModels twice), so the overlay can appear, disappear,
// and REAPPEAR at any point — including mid-sweep — and intercept clicks.
// Dismiss it repeatedly until it stays gone.
async function dismissError(page) {
  const dismiss = page.locator('button.error-dismiss-btn');
  for (let i = 0; i < 5; i++) {
    if (await dismiss.count() === 0) break;
    await dismiss.first().click();
  }
  await expect(dismiss).toHaveCount(0);
}

test('shell renders sidebar nav items', async ({ page }) => {
  await page.goto('/');
  await dismissError(page);
  for (const [key] of NAV_PAGES.slice(0, 10)) {
    await expect(page.getByTestId(`nav-${key}`)).toBeVisible();
  }
});

test('each nav page renders and updates breadcrumb + hash', async ({ page }) => {
  await page.goto('/');
  await dismissError(page);
  for (const [key, label] of NAV_PAGES) {
    const item = page.getByTestId(`nav-${key}`);
    await item.scrollIntoViewIfNeeded();
    // The ErrorUI overlay can reappear mid-sweep and swallow the click
    // (see dismissError). Retry: dismiss, click, and only accept once the
    // hash actually changed.
    await expect(async () => {
      await dismissError(page);
      await item.click({ force: true });
      await expect(page).toHaveURL(new RegExp(`#${key}$`), { timeout: 2000 });
    }).toPass({ timeout: 15_000 });
    await dismissError(page);
    await expect(page.locator('header').getByText(label, { exact: true })).toBeVisible();
  }
});

test('hash-only pages render when opened by URL', async ({ page }) => {
  for (const [key, label] of HASH_ONLY_PAGES) {
    await page.goto(`/#${key}`);
    await dismissError(page);
    await expect(page.locator('header').getByText(label, { exact: true })).toBeVisible();
  }
});

test('activity bar collapses and expands via its toggle button', async ({ page }) => {
  await page.goto('/');
  await dismissError(page);
  const collapsed = page.locator('div.app.activity-collapsed');
  await expect(collapsed).toHaveCount(0);
  const toggle = page.locator('header button[title*="activity bar"]').first();
  await expect(toggle).toBeVisible();
  await toggle.click();
  await expect(collapsed).toHaveCount(1, { timeout: 2000 });
  await page.locator('header button[title*="activity bar"]').first().click();
  await expect(collapsed).toHaveCount(0, { timeout: 2000 });
});

test('topbar shows python status testid', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('python-status')).toBeVisible();
});