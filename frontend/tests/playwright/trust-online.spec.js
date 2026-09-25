const { test, expect } = require('@playwright/test');

// Backend-required trust specs. Run locally with the Python backend up:
//   npx playwright test tests/playwright/trust-online.spec.js
// These tests use the Desktop OS route directory and the real Model Explorer
// load flow; no rendered result is treated as live without a backend payload.

async function loadGpt2(page) {
  await page.goto('/#explorer');
  const explorer = page.getByTestId('window-explorer');
  await expect(explorer.locator('.model-explorer')).toBeVisible({ timeout: 30000 });
  const modelSelect = explorer.locator('#model-explorer-model');
  await expect(modelSelect).toBeVisible({ timeout: 15000 });
  await modelSelect.selectOption('gpt2-small');
  const loadButton = explorer.getByRole('button', { name: /Load model/i });
  await expect(loadButton).toBeEnabled({ timeout: 15000 });
  await loadButton.click();
  await expect(explorer.locator('.loaded-model')).toContainText(/Loaded/i, { timeout: 30000 });
}

async function openCircuitExplorer(page) {
  await page.getByTestId('directory-toggle').click();
  const nav = page.getByTestId('nav-circuitexplorer');
  await nav.scrollIntoViewIfNeeded();
  await nav.click();
  return page.getByTestId('window-circuitexplorer');
}

test('circuit explorer shows live registry circuits', async ({ page }) => {
  await loadGpt2(page);
  const circuitWindow = await openCircuitExplorer(page);

  await expect(circuitWindow.locator('.window-title')).toHaveText('Circuit Explorer');
  const select = circuitWindow.locator('select.input-text');
  await expect(select).toBeVisible({ timeout: 15000 });
  const options = await select.locator('option').allTextContents();
  expect(options.some(text => /Indirect Object Identification/i.test(text))).toBe(true);
  await expect(circuitWindow.getByText(/Cannot reach the circuit registry/)).toHaveCount(0);
});

test('circuit detail shows measured members', async ({ page }) => {
  await loadGpt2(page);
  const circuitWindow = await openCircuitExplorer(page);
  const select = circuitWindow.locator('select.input-text');
  await expect(select).toBeVisible({ timeout: 15000 });
  await expect(circuitWindow.locator('.explorer-page .member-chip').first()).toBeVisible({ timeout: 15000 });
});

test('Research Society runs before model loading and persists its evidence surfaces', async ({ page }) => {
  await page.goto('/#society');
  const society = page.getByTestId('window-society');
  await expect(society.getByRole('heading', { name: 'Research Society', exact: true })).toBeVisible();
  await society.locator('#society-goal').fill('Verify the live Society trace and validation gate contract');
  await society.getByRole('button', { name: 'Run Society' }).click();

  await expect(society).toContainText(/Backend status: completed/i, { timeout: 90000 });
  await expect(society.locator('.society__steps li').first()).toBeVisible({ timeout: 90000 });
  await expect(society.locator('.society__report')).toContainText(/Experiment ID|Generated At/i, { timeout: 90000 });
  await expect(society).toContainText(/Fidelity|Confidence|Validated/i, { timeout: 90000 });
});
