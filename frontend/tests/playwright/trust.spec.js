const { test, expect } = require('@playwright/test');

// Trust UX specs. The API request is deliberately aborted so this test is
// deterministic even when a developer backend is running locally.
test('status indicator reports an unavailable backend without fabricating data', async ({ page }) => {
  await page.route('**/api/**', route => route.abort());
  await page.goto('/');
  const status = page.getByTestId('python-status');
  await expect(status).toBeVisible({ timeout: 15000 });
  await expect(status).toContainText(/Offline/);
});
