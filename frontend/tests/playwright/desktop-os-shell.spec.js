const { test, expect } = require('@playwright/test');

test('deep-linked local tools open without loading a model', async ({ page }) => {
  await page.goto('/#settings');

  const shell = page.getByTestId('app-shell');
  await expect(shell).toBeVisible();
  await expect(page.getByTestId('view-settings')).toBeVisible();
  await expect(page.getByTestId('view-settings').getByRole('heading', { name: 'Settings', exact: true })).toBeVisible();
  await expect(page.getByTestId('runtime-status')).toContainText(/Offline|Connected|models available|No model selected/);
});

test('sidebar navigation switches the visible tool and updates the hash', async ({ page }) => {
  await page.goto('/#explorer');

  await page.getByTestId('nav-society').click();
  await expect(page.getByTestId('view-society')).toBeVisible();
  await expect(page).toHaveURL(/#society$/);

  await page.getByTestId('nav-explorer').click();
  await expect(page.getByTestId('view-explorer')).toBeVisible();
  await expect(page).toHaveURL(/#explorer$/);
});
