const { test, expect } = require('@playwright/test');

test('deep-linked local tools open in Desktop OS without loading a model', async ({ page }) => {
  await page.goto('/#settings');

  const desktop = page.getByTestId('desktop-shell');
  await expect(desktop).toBeVisible();
  await expect(page.getByTestId('window-settings')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Settings' })).toBeVisible();
  await expect(page.getByTestId('python-status')).toContainText(/Offline|Connected/);
});

test('window menu exposes Society and restores a closed tool', async ({ page }) => {
  await page.goto('/#explorer');

  await page.getByRole('button', { name: 'Society' }).click();
  await expect(page.getByTestId('window-society')).toBeVisible();
  await expect(page).toHaveURL(/#society$/);

  await page.getByRole('button', { name: 'Close window' }).click();
  await expect(page.getByTestId('window-society')).toBeHidden();
  await page.getByRole('button', { name: 'Society' }).click();
  await expect(page.getByTestId('window-society')).toBeVisible();
});
