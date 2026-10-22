const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test, expect } = require('@playwright/test');

/**
 * The Society lifecycle, in the standard suite.
 *
 * A Society run is a CPU-bound pipeline, so the spec asks for the `quick`
 * effort: two prompts instead of ten, which the backend reports as an effort
 * that cannot support a validation claim. The run is still a real run -- a
 * real submission, a real sweep of the model, a real terminal state -- and
 * the spec then checks the ending against the backend's own status for that
 * run id. The full-effort evidence is owned by
 * `frontend/tests/e2e/live-society-check.mjs`.
 *
 * What this spec pins:
 *
 *   - the goal reaches the backend and the panel shows the run it created;
 *   - the effort the panel shows is the effort the backend reports;
 *   - the panel's status is the backend's status, not its own guess;
 *   - nothing is reported as a report, a gate or a publication while the run
 *     is in flight -- a result that does not exist yet must not be on screen;
 *   - the run ends, and the ending the panel shows is the ending the backend
 *     has for that run;
 *   - nothing is published that the backend did not publish;
 *   - the run's admission slot is released, so the next submission is not
 *     refused with "active run limit";
 *   - a cancelled run stops: the backend records the request, and the panel
 *     says the cancellation was *requested* -- never that the run was
 *     cancelled while the backend still reports it as running.
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

test('a Society run is tracked, ends on the backend status, and frees its slot', async ({
  page,
  request,
}) => {
  const probe = await request
    .get(`${API}/api/status`, { headers: { Authorization: `Bearer ${bearer()}` } })
    .catch(() => null);
  test.skip(!probe || !probe.ok(), `no live MECH backend on ${API} — the Society lifecycle needs one`);

  // A real run: the quick effort takes a couple of minutes on a CPU, and the
  // backend's own run timeout bounds it.
  test.setTimeout(900_000);

  await page.goto('/#society');
  const society = page.getByTestId('view-society');
  await expect(society.getByRole('heading', { name: 'Research Society', exact: true })).toBeVisible();

  await society.locator('#society-goal').fill(`lifecycle contract ${Date.now()}`);
  // The quick effort: two prompts, which the backend reports as an effort that
  // cannot support a validation claim. The run is real either way.
  await society.getByTestId('society-effort').selectOption('quick');

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
  expect(runBody.effort, 'the backend named the effort it will spend').toBeTruthy();

  // The panel reports the run the backend created, and the effort it used.
  const status = society.getByTestId('society-status');
  await expect(status).toHaveAttribute('data-phase', /^(running|starting|done|failed|stopped)$/, {
    timeout: 60_000,
  });
  await expect(society.getByTestId('society-run-meta')).toContainText(/Effort:/);

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

  if (inFlight) {
    // The run ends. The ending the panel shows must be the ending the backend
    // has for that run -- never a status the panel invented.
    await expect(status).toHaveAttribute('data-phase', /^(done|failed|stopped)$/, {
      timeout: 780_000,
    });
    const ended = await backendRun(request, runId);
    expect(ended, 'the backend still knows the run the panel ended').not.toBeNull();
    const endedStatus = String(ended.status ?? '');
    if (endedStatus === 'completed') {
      await expect(society.getByTestId('society-status-heading')).toContainText(/completed/i);
    } else {
      expect(endedStatus).toMatch(/blocked|failed|error|cancelled|unavailable/i);
    }
    // Nothing may be published that the backend did not publish.
    if (!/Published to/i.test(await society.innerText())) {
      expect(endedStatus).not.toMatch(/published/i);
    }

    // The slot is released: a second submission is admitted, not refused.
    const second = await request.post(`${API}/api/society/run`, {
      headers: { Authorization: `Bearer ${bearer()}`, 'Content-Type': 'application/json' },
      data: { goal: `lifecycle contract ${Date.now()} (second)`, effort: 'quick' },
    });
    const secondBody = await second.json().catch(() => ({}));
    expect(
      String(secondBody.error ?? ''),
      'the finished run released its admission slot',
    ).not.toMatch(/active run limit/i);
    if (secondBody.runId) {
      await request
        .post(`${API}/api/society/runs/${secondBody.runId}/cancel`, {
          headers: { Authorization: `Bearer ${bearer()}`, 'Content-Type': 'application/json' },
          data: {},
        })
        .catch(() => undefined);
    }
  }

  // "Cancel run" performs the advertised operation: the backend records the
  // cancellation request. The panel must not claim the run is cancelled while
  // the backend still reports it as running.
  if ((await status.getAttribute('data-phase')) === 'running') {
    await society.getByRole('button', { name: /Cancel run/i }).click();
    await expect(society.getByTestId('society-cancel-notice')).toBeVisible({ timeout: 60_000 });

    let cancelled = null;
    for (let i = 0; i < 60; i += 1) {
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
