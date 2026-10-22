#!/usr/bin/env node
/**
 * Route sweep: click every sidebar item and record what actually happened.
 *
 * This is the evidence harness for "every sidebar item opens the correct
 * working view". It intentionally runs with whatever backend is (or is not)
 * reachable, and it records console errors, unhandled rejections and failed
 * API calls per route, so a view that renders a fabricated number while the
 * backend is down shows up as a view that produced numbers with no data.
 *
 * Usage:
 *   node tests/e2e/route-sweep.mjs [--base http://localhost:5273] [--out sweep.json]
 *
 * Exit code is 0 when every nav item mounted its own view; 1 otherwise.
 */
import { chromium } from 'playwright';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

const argv = process.argv.slice(2);
const arg = (name, fallback) => {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback;
};

const BASE = arg('--base', 'http://localhost:5273');
const BLOCK_API = argv.includes('--block-api');
const OUT = resolve(arg('--out', 'test-results/route-sweep.json'));
const SHOTS = resolve('test-results/routes');

/** Text that must never appear on a screen while the backend is unavailable. */
const FABRICATION_SIGNS = [
  { id: 'nan', re: /\bNaN\b/ },
  { id: 'undefined', re: /\bundefined\b/ },
  { id: 'infinity', re: /\bInfinity\b/ },
  { id: 'object-object', re: /\[object Object\]/ },
];

mkdirSync(dirname(OUT), { recursive: true });
mkdirSync(SHOTS, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });

if (BLOCK_API) {
  // Reproduces "the backend is gone" without stopping it, so the same sweep
  // proves both the happy path and the honest-failure path.
  await page.route('**/api/**', (route) => route.abort());
}

const consoleErrors = [];
const pageErrors = [];
const failedRequests = [];
const cancelledRequests = [];
const current = { route: null };

page.on('console', (msg) => {
  if (msg.type() !== 'error') return;
  consoleErrors.push({ route: current.route, text: msg.text().slice(0, 400) });
});
page.on('pageerror', (err) => {
  pageErrors.push({ route: current.route, text: String(err).slice(0, 400) });
});
page.on('requestfailed', (req) => {
  const url = req.url();
  if (!url.includes('/api/')) return;
  const error = req.failure()?.errorText ?? '';
  // `net::ERR_ABORTED` is what a view's own AbortController produces when the
  // user navigates away mid-request: that is cancellation working, not failure.
  if (error.includes('ERR_ABORTED')) {
    cancelledRequests.push({ route: current.route, url });
    return;
  }
  failedRequests.push({ route: current.route, url, error });
});
page.on('response', (res) => {
  const url = res.url();
  if (!url.includes('/api/')) return;
  if (res.status() >= 400) {
    failedRequests.push({ route: current.route, url, status: res.status() });
  }
});

await page.goto(BASE, { waitUntil: 'domcontentloaded', timeout: 180_000 });
// `nav-collapse` is the sidebar toggle, not a route: it shares the prefix.
const NAV_SELECTOR = '[data-testid^="nav-"]:not([data-testid="nav-collapse"])';
await page.waitForSelector(NAV_SELECTOR, { timeout: 180_000 });

const navIds = await page.$$eval(NAV_SELECTOR, (nodes) =>
  nodes.map((n) => n.getAttribute('data-testid').slice('nav-'.length)),
);

const results = [];

for (const id of navIds) {
  current.route = id;
  const consoleStart = consoleErrors.length;
  const pageErrorStart = pageErrors.length;
  const reqStart = failedRequests.length;

  const nav = page.locator(`[data-testid="nav-${id}"]`);
  const clicked = await nav.count() > 0;
  let mounted = false;
  let text = '';
  try {
    if (clicked) {
      await nav.first().click({ timeout: 5_000 });
    }
    await page.waitForSelector(`[data-testid="view-${id}"]`, { timeout: 30_000 });
    // The view chunk is async: `view-<id>` is the section wrapper, which appears
    // while the Suspense fallback still reads "Loading <tool>". Wait for the
    // chunk (and the route's first paint) before reading the text.
    await page
      .waitForFunction(
        () => document.querySelectorAll('.route-loading').length === 0,
        undefined,
        { timeout: 90_000 },
      )
      .catch(() => undefined);
    mounted = true;
  } catch {
    mounted = false;
  }

  // Give deferred work (Suspense fallback, first API call, chart paint) a beat.
  await page.waitForTimeout(600 + 100);

  try {
    // Several views render their own `<main>`: scoping to the shell's view
    // wrapper avoids a strict-mode match on multiple landmarks.
    text = (await page
      .locator(`[data-testid="view-${id}"]`)
      .innerText({ timeout: 5_000 }))
      .replace(/\s+/g, ' ')
      .trim();
  } catch {
    text = '';
  }

  const shots = `${SHOTS}/${id}.png`;
  try {
    await page.screenshot({ path: shots, fullPage: false });
  } catch {
    /* screenshot is diagnostic only */
  }

  results.push({
    id,
    clicked,
    mounted,
    hash: await page.evaluate(() => window.location.hash),
    activeNav: await page
      .locator('[data-testid^="nav-"][aria-current="page"]')
      .first()
      .getAttribute('data-testid', { timeout: 2_000 })
      .catch(() => null),
    consoleErrors: consoleErrors.slice(consoleStart),
    pageErrors: pageErrors.slice(pageErrorStart),
    failedRequests: failedRequests.slice(reqStart),
    fabrication: FABRICATION_SIGNS.filter((s) => s.re.test(text)).map((s) => s.id),
    mainLandmarks: await page.locator('main').count(),
    textPreview: text.slice(0, 700),
    screenshot: shots,
  });
}

await browser.close();

const report = {
  base: BASE,
  generatedAt: new Date().toISOString(),
  navCount: navIds.length,
  mounted: results.filter((r) => r.mounted).length,
  notMounted: results.filter((r) => !r.mounted).map((r) => r.id),
  pageErrors: results.filter((r) => r.pageErrors.length).map((r) => r.id),
  fabricationSigns: results.filter((r) => r.fabrication.length).map((r) => ({ id: r.id, signs: r.fabrication })),
  duplicateMainLandmarks: results.filter((r) => r.mainLandmarks > 1).map((r) => ({ id: r.id, main: r.mainLandmarks })),
  results,
};

writeFileSync(OUT, JSON.stringify(report, null, 2));
console.log(JSON.stringify({
  navCount: report.navCount,
  mounted: report.mounted,
  notMounted: report.notMounted,
  pageErrors: report.pageErrors,
  fabricationSigns: report.fabricationSigns,
  failedRequestRoutes: [...new Set(results.filter((r) => r.failedRequests.length).map((r) => r.id))],
  cancelledRequests: cancelledRequests.length,
  out: OUT,
}, null, 2));

process.exit(report.notMounted.length === 0 ? 0 : 1);
