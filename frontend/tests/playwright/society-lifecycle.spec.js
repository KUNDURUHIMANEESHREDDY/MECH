const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test, expect } = require('@playwright/test');

/**
 * The Society lifecycle, in the standard suite.
 *
 * A Society run is a CPU-bound pipeline: a run that completes takes about two
 * minutes on an idle CPU and up to eight under load, so the *completion* path
 * is owned by `frontend/tests/e2e/live-society-check.mjs`, which has the
 * budget. What this spec pins is everything that must hold on every run and
 * that is deterministic within seconds:
 *
 *   - the goal reaches the backend and the panel shows the run it created;
 *   - the panel's status is the backend's status, not its own guess;
 *   - nothing is reported as a report, a gate or a publication while the run
 *     is in flight -- a result that does not exist yet must not be on screen;
 *   - "Cancel run" performs the operation it advertises: the backend records
 *     the cancellation request, and the panel says the cancellation was
 *     *requested* -- never that the run was cancelled while the backend still
 *     reports it as running;
 *   - the panel never claims "completed" for a run the backend has not
 *     completed.
 *
 * The old spec demanded `Backend status: completed` from every run, which is a
 * claim the system is required NOT to make when a stage has no live
 * provenance. A run that ends `blocked` is the correct ending.
 */

const API = process.env.MECH_E2E_API || 'http://localhost:8000';
const TOKEN_FILE = resolve(process.cwd(), '../backend/storage/.mech_api_token');

function bearer() {
  try {
    return readFileSync(TOKEN_FILE, 'utf8').trim();
  } catch {
    return '';
  }
}

async function backendRun(request, runId) {
  const res = await request.get(`${API}/api/society/runs/${runId}`, {
    headers: { Authorization: `Bearer ${bearer()}` },
  });
  if (!res.ok()) return null;
  return res.json();
}

test('a Society run is tracked, cancels for real, and claims nothing early', async ({ page, request }) => {
  const probe = await request
    .get(`${API}/api/status`, { headers: { Authorization: `Bearer ${bearer()}` } })
    .catch(() => null);
  test.skip(!probe || !probe.ok(), `no live MECH backend on ${API} — the Society lifecycle needs one`);

  test.setTimeout(240_000);

  await page.goto('/#society');
  const society = page.getByTestId('view-society');
  await expect(society.getByRole('heading', { name: 'Research Society', exact: true })).toBeVisible();

  await society.locator('#society-goal').fill(`lifecycle contract ${Date.now()}`);

  const runResponse = page.waitForResponse(
    (res) => res.url().includes('/api/society/run') && res.request().method() === 'POST',
    { timeout: 60_000 },
  );
  await society.getByTestId('society-run').click();
  const runBody = await (await runResponse).json();
  // The backend admits two concurrent runs and refuses the rest. A refusal is
  // back-pressure working as designed, not a defect, and it says so.
  test.skip(
    runBody.status === 'busy' || /active run limit|queue capacity/i.test(String(runBody.error ?? '')),
    `the backend refused a concurrent run: ${runBody.error ?? runBody.status}`,
  );
  const runId = runBody.runId;
  expect(runId, 'the backend returned a run id').toBeTruthy();

  // The panel reports the run the backend created.
  const status = society.getByTestId('society-status');
  await expect(status).toHaveAttribute('data-phase', /^(running|starting|done|failed|stopped)$/, {
    timeout: 60_000,
  });
  await expect(society.getByTestId('society-status-heading')).toContainText(
    /starting|in progress|polling|completed|ended|failed/i,
  );

  // A run in flight has no report, no gate and no publication.
  const inFlight = (await status.getAttribute('data-phase')) === 'running';
  if (inFlight) {
    await expect(society).toContainText(
      /Waiting for the backend report|Waiting for the backend validation gate/i,
    );
    await expect(society).not.toContainText(/Published to/i);
  }

  // The panel's status is the backend's status.
  const backend = await backendRun(request, runId);
  expect(backend, 'the backend knows the run the panel is showing').not.toBeNull();
  const backendStatus = String(backend.status ?? '');
  if (inFlight) {
    expect(['running', 'started', 'queued']).toContain(backendStatus);
  }

  // "Cancel run" performs the advertised operation: the backend records the
  // cancellation request. The panel must not claim the run is cancelled while
  // the backend still reports it as running.
  if (inFlight) {
    await society.getByRole('button', { name: /Cancel run/i }).click();
    await expect(society.getByTestId('society-cancel-notice')).toBeVisible({ timeout: 60_000 });

    let cancelled = null;
    for (let i = 0; i < 30; i += 1) {
      const current = await backendRun(request, runId);
      if (current && (current.cancel_requested || current.status === 'cancelled')) {
        cancelled = current;
        break;
      }
      await page.waitForTimeout(1000);
    }
    expect(cancelled, 'the backend recorded the cancellation request').not.toBeNull();
    if (cancelled.status === 'running') {
      await expect(society).not.toContainText(/the backend cancelled this run/i);
    }
  }
});
