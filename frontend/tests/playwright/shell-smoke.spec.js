const { test, expect } = require('@playwright/test');

// Canonical route labels. Hash IDs remain compatible with deep links.
//
// This is every route the shell renders a nav button for. The three alias
// routes (gpt2, gpt2explorer, campaigns) are deliberately absent: the
// registry marks them `aliasOf` a canonical route, and the shell renders one
// nav entry per canonical route so a keyboard or screen-reader user is never
// offered the same tool twice. This list still expected `nav-gpt2` and
// `nav-campaigns`, which the shell has not rendered since the aliases were
// introduced, so the test failed on its second entry and stopped there.
const NAV_PAGES = [
  ['explorer', 'Model Explorer'],
  ['transformer', 'Transformer Visualizer'],
  ['transformerExplorer', 'Transformer Explorer'],
  ['network', 'Network'],
  ['steering', 'Steering Lab'],
  ['workspace', 'Workspace'],
  ['models', 'Models'],
  ['prompts', 'Prompts'],
  ['debugger', 'Debugger'],
  ['experiments', 'Experiments'],
  ['sessions', 'Sessions'],
  ['reports', 'Reports'],
  ['settings', 'Settings'],
  ['logging', 'Logging'],
  ['build', 'Build'],
  ['neuralexplorer', 'Neural Explorer'],
  ['benchmark', 'Benchmark'],
  ['benchmarksuite', 'Benchmark Suite'],
  ['knowledgegraph', 'Knowledge Graph'],
  ['circuitexplorer', 'Circuit Explorer'],
  ['reasoning', 'Reasoning'],
  ['evidencefusion', 'Evidence Fusion'],
  ['analytics', 'Analytics'],
  ['health', 'Health'],
  ['plugins', 'Plugins'],
  ['notebook', 'Research Notebook'],
  ['labnotebook', 'Lab Notebook'],
  ['reproduction', 'Paper Reproduction'],
  ['projects', 'Projects'],
  ['recent', 'Recent Files'],
  ['society', 'Society'],
  ['activations', 'Activations'],
  ['logitlens', 'Logit Lens'],
  ['saelab', 'SAE Lab'],
  ['datasets', 'Datasets'],
];

// Routes with no nav button of their own: they open by hash, and while one
// is open the shell highlights the canonical entry it aliases.
const HASH_ONLY_PAGES = [
  ['gpt2', 'GPT-2 Live'],
  ['gpt2explorer', 'GPT-2 Neuron Explorer'],
  ['campaigns', 'Campaigns'],
];

test('sidebar exposes every navigation item', async ({ page }) => {
  await page.goto('/');
  for (const [key] of NAV_PAGES) {
    await expect(page.getByTestId(`nav-${key}`)).toBeVisible();
  }
});

test('sidebar navigation shows one titled tool view and updates the hash', async ({ page }) => {
  // One click per route, each with a scroll, a URL assertion and two view
  // assertions: 36 routes, each with its own assertion budget, and a test
  // budget that covers the whole list.
  test.setTimeout(1200_000);
  await page.goto('/');
  for (const [key] of NAV_PAGES) {
    const item = page.getByTestId(`nav-${key}`);
    await item.scrollIntoViewIfNeeded();
    await item.click();
    await expect(page).toHaveURL(new RegExp(`#${key}$`));
    const view = page.getByTestId(`view-${key}`);
    // A route's chunk is compiled on first request, so the first visit to a
    // tool can take longer than the default per-assertion budget on a
    // loaded machine.
    await expect(view).toBeVisible({ timeout: 30_000 });
    await expect(view.getByRole('heading').first()).toBeVisible({ timeout: 30_000 });
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
  await expect(status).toContainText(/Connecting to backend|Backend ready|Disconnected|Backend failed|Unauthorized/);
});
