const { test, expect } = require('@playwright/test');

test('local tool windows expose honest empty states when the backend is offline', async ({ page }) => {
  await page.route('**/api/**', route => route.abort());
  await page.goto('/#projects');
  const projects = page.getByTestId('window-projects');
  await expect(projects).toBeVisible();
  await expect(projects.locator('.tool-window__body').getByRole('heading', { name: 'Projects', exact: true })).toBeVisible();
  await expect(projects).toContainText(/No projects|unavailable|offline/i);

  await page.goto('/#reports');
  const reports = page.getByTestId('window-reports');
  await expect(reports).toBeVisible();
  await expect(reports).toContainText(/No reports|unavailable|offline/i);
});

test('model and research tools remain reachable without the old global gate', async ({ page }) => {
  await page.goto('/#models');
  await expect(page.getByTestId('window-models')).toBeVisible();
  await expect(page.getByTestId('window-models').locator('.tool-window__body').getByRole('heading', { name: 'Models', exact: true })).toBeVisible();

  await page.goto('/#workspace');
  await expect(page.getByTestId('window-workspace')).toBeVisible();
  await expect(page.getByTestId('window-workspace')).toContainText(/Research workspace|Workspace/i);
});
