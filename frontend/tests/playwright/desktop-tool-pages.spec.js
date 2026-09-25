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

test('remaining legacy tools expose real surfaces instead of placeholders', async ({ page }) => {
  await page.route('**/api/**', route => route.abort());
  const routes = [
    ['models', 'Models', /model records|no models|unavailable/i],
    ['prompts', 'Prompts', /local prompt drafts|unavailable|no prompts/i],
    ['debugger', 'Debugger', /diagnos|unavailable|offline/i],
    ['experiments', 'Experiments', /experiment records|unavailable|offline/i],
    ['sessions', 'Sessions', /session records|unavailable|offline/i],
    ['plugins', 'Plugins', /plugin catalog|appapi|unavailable/i],
    ['notebook', 'Research Notebook', /research notes|local-only|unavailable/i],
    ['labnotebook', 'Lab Notebook', /lab notes|local-only|unavailable/i],
    ['transformer', 'Transformer Visualizer', /architecture|offline|unavailable/i],
  ];

  for (const [id, heading, marker] of routes) {
    await page.goto(`/#${id}`);
    const window = page.getByTestId(`window-${id}`);
    await expect(window).toBeVisible();
    await expect(window.locator('.tool-window__body').getByRole('heading', { name: heading, exact: true })).toBeVisible();
    await expect(window.locator('.tool-window__body')).toContainText(marker);
  }
});
