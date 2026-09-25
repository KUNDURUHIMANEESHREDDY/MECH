const { test, expect } = require('@playwright/test');

// Canonical Desktop OS route labels. The hash IDs remain compatible with
// the previous Vue shell and its deep links.
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

async function openDirectory(page) {
  const directory = page.getByTestId('window-directory');
  if (!(await directory.isVisible())) {
    await page.getByTestId('directory-toggle').click();
  }
  await expect(directory).toBeVisible();
}

test('Desktop OS directory exposes legacy navigation items', async ({ page }) => {
  await page.goto('/');
  await openDirectory(page);
  for (const [key] of NAV_PAGES) {
    await expect(page.getByTestId(`nav-${key}`)).toBeVisible();
  }
});

test('directory navigation opens a titled Desktop OS window and updates the hash', async ({ page }) => {
  await page.goto('/');
  for (const [key, label] of NAV_PAGES) {
    await openDirectory(page);
    const item = page.getByTestId(`nav-${key}`);
    await item.scrollIntoViewIfNeeded();
    await item.click();
    await expect(page).toHaveURL(new RegExp(`#${key}$`));
    await expect(page.getByTestId(`window-${key}`)).toBeVisible();
    await expect(page.getByTestId(`window-${key}`).locator('.window-title')).toHaveText(label);
  }
});

test('hash-only pages open without a model gate', async ({ page }) => {
  for (const [key, label] of HASH_ONLY_PAGES) {
    await page.goto(`/#${key}`);
    await expect(page.getByTestId(`window-${key}`)).toBeVisible();
    await expect(page.getByTestId(`window-${key}`).locator('.window-title')).toHaveText(label);
  }
});

test('Desktop OS window controls tile, minimize and close tools', async ({ page }) => {
  await page.goto('/#explorer');
  await page.getByRole('button', { name: 'Society' }).click();
  await expect(page.getByTestId('window-society')).toBeVisible();

  await page.getByRole('button', { name: 'Tile windows' }).click();
  await expect(page.getByTestId('window-explorer')).toBeVisible();
  await expect(page.getByTestId('window-society')).toBeVisible();

  await page.getByTestId('window-society').getByRole('button', { name: 'Minimize window' }).click();
  await expect(page.getByTestId('window-society')).toHaveCount(0);

  await page.getByRole('button', { name: 'Society' }).click();
  await expect(page.getByTestId('window-society')).toBeVisible();
  await page.getByTestId('window-society').getByRole('button', { name: 'Close window' }).click();
  await expect(page.getByTestId('window-society')).toHaveCount(0);
});

test('desktop status exposes Python/runtime state', async ({ page }) => {
  await page.goto('/');
  const status = page.getByTestId('python-status');
  await expect(status).toBeVisible();
  await expect(status).toContainText(/Connecting|Connected|Offline/);
});
