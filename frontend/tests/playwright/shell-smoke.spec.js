const { test, expect } = require('@playwright/test');

// Labels must match App.tsx Shell PAGE_LABEL_MAP (source of the Topbar crumb).
// navKeys mirror the FeaturesDrawer RESOURCE_TREE.
const NAV_PAGES = [
  ['active_investigation', 'Overview (Active Investigation)'],
  ['models', 'Models'],
  ['dataset_viewer', 'Datasets & Probes'],
  ['hypothesis_lab', 'Hypothesis Lab & Falsification'],
  ['mechanism_builder', 'Visual Mechanism Builder'],
  ['evidence_graph', 'Evidence Graph & Provenance'],
  ['report_mode', 'Research Artifacts & Reports'],
  ['model_explorer', 'Model Architecture Explorer'],
  ['transformer', 'Transformer Visualizer'],
  ['attention_heatmap', 'Attention Lab'],
  ['neuron_panel', 'Neuron & MLP Explorer'],
  ['sae_feature', 'SAE Feature Explorer'],
  ['circuit_explorer', 'Circuit Explorer'],
  ['logit_lens', 'Logit Lens & Trajectory'],
  ['intervention_lab', 'Causal Intervention Lab'],
  ['experiment_builder', 'Experiment Builder'],
  ['experiment_monitor', 'Experiment Monitor'],
  ['evidence_explorer', 'Evidence Explorer'],
  ['research_graph', 'Research Graph'],
  ['research_notebook', 'Research Notebook'],
  ['experiment_notebook', 'Research Notebook'],
  ['health', 'Reproducibility & Validation'],
  ['compute_center', 'Compute Center & Jobs'],
  ['real_time_dag', 'Storage & Telemetry DAG'],
  ['logging', 'Execution Logs'],
  ['projects', 'Projects'],
  ['settings', 'Settings'],
  ['ui_adversarial_tester', 'UI Adversarial Tester'],
  ['acceptance_test_verifier', 'Acceptance Test Verifier'],
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
  await page.getByTestId('features-toggle').click();
  for (const [key] of NAV_PAGES.slice(0, 10)) {
    await expect(page.getByTestId(`nav-${key}`).first()).toBeVisible();
  }
});

test('features drawer opens and closes via its toggle button', async ({ page }) => {
  await page.goto('/');
  await dismissError(page);
  const drawer = page.getByTestId('features-drawer');
  const toggle = page.getByTestId('features-toggle');
  await expect(drawer).toHaveCount(0);
  await toggle.click();
  await expect(drawer).toHaveCount(1, { timeout: 2000 });
  await toggle.click();
  await expect(drawer).toHaveCount(0, { timeout: 2000 });
});

test('each nav page renders and updates breadcrumb + hash', async ({ page }) => {
  await page.goto('/');
  await dismissError(page);
  await page.getByTestId('features-toggle').click();
  for (const [key, label] of NAV_PAGES) {
    const item = page.getByTestId(`nav-${key}`).first();
    await item.scrollIntoViewIfNeeded();
    // The ErrorUI overlay can reappear mid-sweep and swallow the click
    // (see dismissError). Retry: dismiss, click, and only accept once the
    // hash actually changed.
    await expect(async () => {
      await dismissError(page);
      await item.scrollIntoViewIfNeeded();
      await page.waitForTimeout(250);
      await item.click({ force: true });
      await page.waitForTimeout(350);
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