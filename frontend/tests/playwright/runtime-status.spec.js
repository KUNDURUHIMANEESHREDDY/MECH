const { test, expect } = require('@playwright/test');

// The sidebar pill is the app's only global claim about the backend. It used to
// be driven by `GET /api/models` and rendered "N models available" on success,
// which turned a static reference catalogue into a claim of research
// capability. These specs pin the four documented states
// (src/services/runtimeStatus.ts) and the retry control that replaces guessing.
//
// Every response is intercepted, so the specs are deterministic with or without
// a developer backend running on :8000.

const pill = (page) => page.getByTestId('runtime-status');

test('a backend that never answers reports failed, not ready', async ({ page }) => {
  await page.route('**/api/status', (route) => route.abort());
  await page.goto('/');

  await expect(pill(page)).toContainText(/Backend failed/i, { timeout: 15000 });
  await expect(pill(page).getByTestId('runtime-retry')).toBeVisible();
  await expect(pill(page)).not.toContainText(/models available/i);
});

test('an answered probe reports ready and makes no model claim', async ({ page }) => {
  await page.route('**/api/status', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'ok' }) }),
  );
  await page.goto('/');

  await expect(pill(page)).toContainText(/Backend ready/i, { timeout: 15000 });
  // `ready` is a claim about the control plane only: no count of models, no
  // "loaded" wording, and no retry offered while the API answers.
  await expect(pill(page)).not.toContainText(/models available|loaded/i);
  await expect(pill(page).getByTestId('runtime-retry')).toHaveCount(0);
});

test('a backend that stops answering after answering reports disconnected', async ({ page }) => {
  let answering = true;
  await page.route('**/api/status', (route) => {
    if (answering) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'ok' }),
      });
    }
    return route.abort();
  });

  await page.goto('/');
  await expect(pill(page)).toContainText(/Backend ready/i, { timeout: 15000 });

  answering = false;
  // The shell re-probes when the window regains attention, which is the only
  // way a drop can be observed: reporting it as `failed` would claim the
  // backend never answered, and this session saw it answer.
  await page.evaluate(() => window.dispatchEvent(new Event('focus')));
  await expect(pill(page)).toContainText(/Disconnected/i, { timeout: 15000 });
});

test('a refused token reports unauthorized rather than unreachable', async ({ page }) => {
  await page.route('**/api/status', (route) => route.fulfill({ status: 401, body: 'nope' }));
  await page.goto('/');

  await expect(pill(page)).toContainText(/Unauthorized/i, { timeout: 15000 });
  // The HTTP status lives in the detail (title), not the one-line label.
  await expect(pill(page).locator('.nav-status__text')).toHaveAttribute('title', /401/);
});

test('retry re-probes and reflects recovery', async ({ page }) => {
  let answering = false;
  await page.route('**/api/status', (route) => {
    if (!answering) return route.abort();
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'ok' }),
    });
  });

  await page.goto('/');
  await expect(pill(page)).toContainText(/Backend failed/i, { timeout: 15000 });

  answering = true;
  await pill(page).getByTestId('runtime-retry').click();
  await expect(pill(page)).toContainText(/Backend ready/i, { timeout: 15000 });
});

test('retry is keyboard reachable', async ({ page }) => {
  await page.route('**/api/status', (route) => route.abort());
  await page.goto('/');
  await expect(pill(page)).toContainText(/Backend failed/i, { timeout: 15000 });

  const retry = pill(page).getByTestId('runtime-retry');
  await expect(retry).toBeEnabled();
  await retry.focus();
  await expect(retry).toBeFocused();
});
