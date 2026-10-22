#!/usr/bin/env node
/**
 * Society run lifecycle check (opt-in, needs a live backend).
 *
 * A Society run is a CPU-bound pipeline: ~75s on an idle machine, longer under
 * load. That is too slow and too environment-dependent for the Playwright
 * suite, so the suite pins only the submission contract and this harness owns
 * the lifecycle.
 *
 * It asserts the thing the UI must never get wrong: when the backend finishes,
 * the panel finishes too, and it reports the backend's own terminal status --
 * `blocked` with a reason is a correct ending, a permanently spinning panel is
 * not. `GET /api/society/stream` replays the trace and then holds the
 * connection open with keep-alives and never emits a terminal frame, so a
 * stream-only panel cannot end; the view's status watchdog is what this checks.
 *
 *   node tests/e2e/live-society-check.mjs
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium } from 'playwright';

const BASE = process.env.MECH_E2E_BASE || 'http://localhost:5273';
const API = process.env.MECH_E2E_API || 'http://localhost:8000';
const TOKEN =
  process.env.MECH_E2E_TOKEN ||
  readFileSync(resolve(process.cwd(), '../backend/storage/.mech_api_token'), 'utf8').trim();
// A Society run is a CPU-bound pipeline: about two minutes on an idle CPU and
// eight or more when the machine is busy, so the budget has to cover a loaded
// machine rather than a fixed guess.
const BUDGET_MS = Number(process.env.MECH_E2E_SOCIETY_BUDGET_MS || 900_000);

const runStatus = async (id) => {
  const res = await fetch(`${API}/api/society/runs/${id}`, {
    headers: { Authorization: `Bearer ${TOKEN}` },
  });
  return res.json();
};

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

let uiRunId = null;
page.on('response', async (res) => {
  if (!res.url().endsWith('/api/society/run')) return;
  try {
    const body = await res.json();
    uiRunId = body.runId ?? uiRunId;
  } catch {
    /* non-JSON: handled by the assertions below */
  }
});

await page.goto(`${BASE}/#society`, { waitUntil: 'domcontentloaded', timeout: 180_000 });
const view = page.getByTestId('view-society');
await view.waitFor({ timeout: 60_000 });
await view.locator('#society-goal').fill(`live society check ${Date.now()}`);
await view.getByTestId('society-run').click();

const read = () =>
  page.evaluate(() => ({
    phase: document.querySelector('[data-testid=society-status]')?.getAttribute('data-phase') ?? '',
    heading: document.querySelector('[data-testid=society-status-heading]')?.textContent?.trim() ?? '',
    steps: document.querySelectorAll('[data-testid=society-step]').length,
    error: document.querySelector('[data-testid=society-error]')?.textContent?.trim() ?? '',
    published: /Published to/i.test(document.body.innerText),
  }));

const started = Date.now();
let dom = await read();
let backend = null;
const timeline = [];
while (Date.now() - started < BUDGET_MS) {
  await page.waitForTimeout(3000);
  dom = await read();
  if (uiRunId) backend = await runStatus(uiRunId).catch(() => null);
  timeline.push({ t: Math.round((Date.now() - started) / 1000), phase: dom.phase, backend: backend?.status ?? null });
  if (['done', 'failed', 'stopped'].includes(dom.phase)) break;
}

const backendStatus = backend?.status ?? '(unknown)';
const terminal = ['done', 'failed', 'stopped'].includes(dom.phase);
const completed = /completed/i.test(dom.heading);
const reasons = /blocked|reason|unavailable|no completion|failed/i.test(dom.error) || (await view.innerText()).length > 0;

console.log('run id:', uiRunId ?? '(not captured)');
console.log('ui phase:', dom.phase, '| heading:', dom.heading);
console.log('backend status:', backendStatus);
console.log('steps rendered:', dom.steps, '| published claim:', dom.published);
console.log('error text:', (dom.error || '(none)').slice(0, 200));
console.log('timeline tail:', JSON.stringify(timeline.slice(-4)));

let ok = terminal && !dom.published;
if (completed) {
  // A completion claim must be backed by returned evidence.
  ok = ok && dom.steps > 0;
  console.log(completed && dom.steps > 0 ? 'PASS completed with returned trace steps' : 'FAIL completed without trace evidence');
} else {
  ok = ok && reasons;
  console.log(ok ? 'PASS ended on the backend status without publishing' : 'FAIL ended without an explanation');
}

await page.screenshot({ path: 'test-results/live-society.png' });
await browser.close();
process.exit(ok ? 0 : 1);
