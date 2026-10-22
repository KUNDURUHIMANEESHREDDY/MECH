const { test, expect } = require('@playwright/test');

// Trust UX specs. The API request is deliberately aborted so this test is
// deterministic even when a developer backend is running locally.
test('status indicator reports an unavailable backend without fabricating data', async ({ page }) => {
  await page.route('**/api/**', route => route.abort());
  await page.goto('/');
  const status = page.getByTestId('runtime-status');
  await expect(status).toBeVisible({ timeout: 15000 });
  // Four-state vocabulary (src/services/runtimeStatus.ts): a backend that never
  // answered is `failed`, and the pill offers a retry instead of guessing.
  await expect(status).toContainText(/Backend failed/i);
  await expect(status.getByTestId('runtime-retry')).toBeVisible();
  await expect(status).not.toContainText(/models available/i);
});
