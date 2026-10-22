const { test, expect } = require('@playwright/test');

// Backend-required trust specs. Run locally with the Python backend up:
//   npx playwright test tests/playwright/trust-online.spec.js
// These tests use the sidebar navigation and the real Model Explorer
// load flow; no rendered result is treated as live without a backend payload.

async function loadGpt2(page) {
  await page.goto('/#explorer');
  const explorer = page.getByTestId('view-explorer');
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
  const nav = page.getByTestId('nav-circuitexplorer');
  await nav.scrollIntoViewIfNeeded();
  await nav.click();
  return page.getByTestId('view-circuitexplorer');
}

test('circuit explorer shows live registry circuits', async ({ page }) => {
  await loadGpt2(page);
  const circuitView = await openCircuitExplorer(page);

  await expect(circuitView.getByRole('heading', { name: 'Circuit Explorer' }).first()).toBeVisible();
  const select = circuitView.locator('select.input-text');
  await expect(select).toBeVisible({ timeout: 15000 });
  const options = await select.locator('option').allTextContents();
  expect(options.some(text => /Indirect Object Identification/i.test(text))).toBe(true);
  await expect(circuitView.getByText(/Cannot reach the circuit registry/)).toHaveCount(0);
});

test('circuit detail shows measured members', async ({ page }) => {
  await loadGpt2(page);
  const circuitView = await openCircuitExplorer(page);
  const select = circuitView.locator('select.input-text');
  await expect(select).toBeVisible({ timeout: 15000 });
  await expect(circuitView.locator('.explorer-page .member-chip').first()).toBeVisible({ timeout: 15000 });
});

// The Society lifecycle is owned by society-lifecycle.spec.js, which is the
// only spec that starts a Society run: the backend admits two concurrent runs,
// so two specs starting one each would make both of them race for the same
// slots. That spec also pins the "no result before the backend returns one"
// rule this file used to check.

test('native Vue visualization surfaces mount without a React bridge error', async ({ page }) => {
  for (const id of ['neuralexplorer', 'transformerExplorer']) {
    await page.goto(`/#${id}`);
    const view = page.getByTestId(`view-${id}`);
    await expect(view).toBeVisible({ timeout: 30000 });
    await expect(view.locator('.route-error')).toHaveCount(0);
    await expect(view).toContainText(/Layer|Attention|Model/i);
  }
});
