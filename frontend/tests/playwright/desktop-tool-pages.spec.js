const { test, expect } = require('@playwright/test');

test('local tool views expose honest empty states when the backend is offline', async ({ page }) => {
  await page.route('**/api/**', route => route.abort());
  await page.goto('/#projects');
  const projects = page.getByTestId('view-projects');
  await expect(projects).toBeVisible();
  await expect(projects.getByRole('heading', { name: 'Projects', exact: true })).toBeVisible();
  await expect(projects).toContainText(/No projects|unavailable|offline/i);

  await page.goto('/#reports');
  const reports = page.getByTestId('view-reports');
  await expect(reports).toBeVisible();
  await expect(reports).toContainText(/No reports|unavailable|offline/i);
});

test('model and research tools remain reachable without the old global gate', async ({ page }) => {
  await page.goto('/#models');
  await expect(page.getByTestId('view-models')).toBeVisible();
  await expect(page.getByTestId('view-models').getByRole('heading', { name: 'Models', exact: true })).toBeVisible();

  await page.goto('/#workspace');
  await expect(page.getByTestId('view-workspace')).toBeVisible();
  await expect(page.getByTestId('view-workspace')).toContainText(/Research workspace|Workspace/i);
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
    const view = page.getByTestId(`view-${id}`);
    await expect(view).toBeVisible();
    await expect(view.getByRole('heading', { name: heading, exact: true })).toBeVisible();
    await expect(view).toContainText(marker);
  }
});
