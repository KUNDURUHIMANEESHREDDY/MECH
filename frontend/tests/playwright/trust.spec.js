const { test, expect } = require('@playwright/test');

// Trust UX specs. These run WITHOUT the Python backend (same offline
// assumption as shell-smoke): the app must say so instead of faking data.

test('status indicator shows offline without backend', async ({ page }) => {
  await page.goto('/');
  const status = page.getByTestId('python-status');
  await expect(status).toBeVisible({ timeout: 15000 });
  await expect(status).toContainText(/Offline/);
});

// NOTE: the circuit explorer view sits behind the model-load gate, so its
// backend states are covered by trust-online.spec.js (requires backend).
// This file stays offline-safe for CI.
