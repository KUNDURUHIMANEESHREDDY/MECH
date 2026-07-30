// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('DesktopApp renderer', () => {
  test('sidebar navigation works and settings page renders tabs', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByTestId('nav-workspace')).toBeVisible();
    await expect(page.getByTestId('nav-settings')).toBeVisible();

    await page.getByTestId('nav-settings').click();
    await expect(page.getByTestId('settings-tab-theme')).toBeVisible();
    await expect(page.getByTestId('settings-tab-gpu')).toBeVisible();
    await expect(page.getByTestId('settings-tab-cache')).toBeVisible();
    await expect(page.getByTestId('settings-tab-paths')).toBeVisible();
  });

  test('theme select reflects current settings', async ({ page }) => {
    await page.goto('/');
    await page.getByTestId('nav-settings').click();
    const select = page.getByTestId('theme-select');
    await expect(select).toBeVisible();
    await select.selectOption('dark');
    // theme attribute is applied immediately by the App component
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  });

  test('projects page lets you add a project', async ({ page }) => {
    await page.goto('/');
    await page.getByTestId('nav-projects').click();
    await page.locator('input').first().fill('E2E Project');
    await page.getByRole('button', { name: 'Add project' }).click();
    await expect(page.getByTestId('projects-list')).toContainText('E2E Project');
  });
});
