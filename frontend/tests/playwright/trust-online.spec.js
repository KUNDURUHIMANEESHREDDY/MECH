const { test, expect } = require('@playwright/test');

// Backend-required trust specs. NOT run in CI (no backend there) — run
// locally with the Python backend up: npx playwright test trust-online.spec.js
// Proves the full vertical: registry -> view -> members, with zero fakes.

async function loadModel(page) {
  await page.goto('/');
  const loadBtn = page.getByRole('button', { name: /Load gpt/i });
  await expect(loadBtn.first()).toBeVisible({ timeout: 15000 });
  await loadBtn.first().click();
}

test('circuit explorer shows live registry circuits', async ({ page }) => {
  await loadModel(page);
  const nav = page.getByTestId('nav-circuitexplorer');
  await nav.scrollIntoViewIfNeeded();
  await nav.click();
  await expect(page.locator('header').getByText('Circuit Explorer', { exact: true })).toBeVisible();
  const select = page.locator('.explorer-page select.input-text');
  await expect(select).toBeVisible({ timeout: 15000 });
  const options = await select.locator('option').allTextContents();
  expect(options.some(t => /Indirect Object Identification/i.test(t))).toBe(true);
  // No backend error state when the registry is reachable.
  await expect(page.getByText(/Cannot reach the circuit registry/)).toHaveCount(0);
});

test('circuit detail shows measured members', async ({ page }) => {
  await loadModel(page);
  const nav = page.getByTestId('nav-circuitexplorer');
  await nav.scrollIntoViewIfNeeded();
  await nav.click();
  const select = page.locator('.explorer-page select.input-text');
  await expect(select).toBeVisible({ timeout: 15000 });
  // Detail panel renders member chips from the live registry payload.
  await expect(page.locator('.explorer-page .member-chip').first()).toBeVisible({ timeout: 15000 });
});
