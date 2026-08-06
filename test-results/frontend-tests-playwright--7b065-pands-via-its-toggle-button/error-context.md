# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: frontend\tests\playwright\shell-smoke.spec.js >> activity bar collapses and expands via its toggle button
- Location: frontend\tests\playwright\shell-smoke.spec.js:91:1

# Error details

```
Error: page.goto: Protocol error (Page.navigate): Cannot navigate to invalid URL
Call log:
  - navigating to "/", waiting until "load"

```

# Test source

```ts
  1   | const { test, expect } = require('@playwright/test');
  2   | 
  3   | // Labels must match App.tsx PAGES (source of the Topbar breadcrumb).
  4   | const NAV_PAGES = [
  5   |   ['explorer', 'Model Explorer'],
  6   |   ['gpt2', 'GPT-2 Live'],
  7   |   ['transformer', 'Transformer Visualizer'],
  8   |   ['workspace', 'Workspace'],
  9   |   ['neuralexplorer', 'Neural Explorer'],
  10  |   ['knowledgegraph', 'Knowledge Graph'],
  11  |   ['circuitexplorer', 'Circuit Explorer'],
  12  |   ['models', 'Models'],
  13  |   ['prompts', 'Prompts'],
  14  |   ['debugger', 'Debugger'],
  15  |   ['build', 'Build'],
  16  |   ['benchmark', 'Benchmark'],
  17  |   ['benchmarksuite', 'Benchmark Suite'],
  18  |   ['experiments', 'Experiments'],
  19  |   ['sessions', 'Sessions'],
  20  |   ['reports', 'Reports'],
  21  |   ['reasoning', 'Reasoning'],
  22  |   ['evidencefusion', 'Evidence Fusion'],
  23  |   ['analytics', 'Analytics'],
  24  |   ['health', 'Health'],
  25  |   ['campaigns', 'Campaigns'],
  26  |   ['plugins', 'Plugins'],
  27  |   ['notebook', 'Research Notebook'],
  28  |   ['labnotebook', 'Lab Notebook'],
  29  |   ['reproduction', 'Paper Reproduction'],
  30  |   ['settings', 'Settings'],
  31  |   ['logging', 'Logging'],
  32  |   ['projects', 'Projects'],
  33  |   ['recent', 'Recent Files'],
  34  | ];
  35  | 
  36  | const HASH_ONLY_PAGES = [
  37  |   ['gpt2explorer', 'GPT-2 Neuron Explorer'],
  38  |   ['transformerExplorer', 'Transformer Explorer'],
  39  | ];
  40  | 
  41  | // The renderer runs standalone in these e2e tests (no Python backend at
  42  | // localhost:8000), so listModels() rejects and ErrorUI overlays the app.
  43  | // The rejection is async relative to page.goto (and the dev-mode StrictMode
  44  | // double-mount fires listModels twice), so the overlay can appear, disappear,
  45  | // and REAPPEAR at any point — including mid-sweep — and intercept clicks.
  46  | // Dismiss it repeatedly until it stays gone.
  47  | async function dismissError(page) {
  48  |   const dismiss = page.locator('button.error-dismiss-btn');
  49  |   for (let i = 0; i < 5; i++) {
  50  |     if (await dismiss.count() === 0) break;
  51  |     await dismiss.first().click();
  52  |   }
  53  |   await expect(dismiss).toHaveCount(0);
  54  | }
  55  | 
  56  | test('shell renders sidebar nav items', async ({ page }) => {
  57  |   await page.goto('/');
  58  |   await dismissError(page);
  59  |   for (const [key] of NAV_PAGES.slice(0, 10)) {
  60  |     await expect(page.getByTestId(`nav-${key}`)).toBeVisible();
  61  |   }
  62  | });
  63  | 
  64  | test('each nav page renders and updates breadcrumb + hash', async ({ page }) => {
  65  |   await page.goto('/');
  66  |   await dismissError(page);
  67  |   for (const [key, label] of NAV_PAGES) {
  68  |     const item = page.getByTestId(`nav-${key}`);
  69  |     await item.scrollIntoViewIfNeeded();
  70  |     // The ErrorUI overlay can reappear mid-sweep and swallow the click
  71  |     // (see dismissError). Retry: dismiss, click, and only accept once the
  72  |     // hash actually changed.
  73  |     await expect(async () => {
  74  |       await dismissError(page);
  75  |       await item.click({ force: true });
  76  |       await expect(page).toHaveURL(new RegExp(`#${key}$`), { timeout: 2000 });
  77  |     }).toPass({ timeout: 15_000 });
  78  |     await dismissError(page);
  79  |     await expect(page.locator('header').getByText(label, { exact: true })).toBeVisible();
  80  |   }
  81  | });
  82  | 
  83  | test('hash-only pages render when opened by URL', async ({ page }) => {
  84  |   for (const [key, label] of HASH_ONLY_PAGES) {
  85  |     await page.goto(`/#${key}`);
  86  |     await dismissError(page);
  87  |     await expect(page.locator('header').getByText(label, { exact: true })).toBeVisible();
  88  |   }
  89  | });
  90  | 
  91  | test('activity bar collapses and expands via its toggle button', async ({ page }) => {
> 92  |   await page.goto('/');
      |              ^ Error: page.goto: Protocol error (Page.navigate): Cannot navigate to invalid URL
  93  |   await dismissError(page);
  94  |   const collapsed = page.locator('div.app.activity-collapsed');
  95  |   await expect(collapsed).toHaveCount(0);
  96  |   const toggle = page.locator('header button[title*="activity bar"]').first();
  97  |   await expect(toggle).toBeVisible();
  98  |   await toggle.click();
  99  |   await expect(collapsed).toHaveCount(1, { timeout: 2000 });
  100 |   await page.locator('header button[title*="activity bar"]').first().click();
  101 |   await expect(collapsed).toHaveCount(0, { timeout: 2000 });
  102 | });
  103 | 
  104 | test('topbar shows python status testid', async ({ page }) => {
  105 |   await page.goto('/');
  106 |   await expect(page.getByTestId('python-status')).toBeVisible();
  107 | });
```