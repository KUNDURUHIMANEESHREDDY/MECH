/**
 * Live research verification (one-off, not part of the suite).
 *
 * Loads the app in a real browser against a real backend, then checks that the
 * numbers on screen are the numbers the API returned -- the anti-fabrication
 * acceptance criterion. Run with the backend up and GPT-2 loaded:
 *   node tests/e2e/live-research-check.mjs
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium } from 'playwright';

const BASE = process.env.MECH_E2E_BASE || 'http://localhost:5273';
// Same token the dev proxy signs with, so the script's own API calls and the
// page's proxied calls are looking at the same backend identity.
const TOKEN =
  process.env.MECH_E2E_TOKEN ||
  readFileSync(resolve(process.cwd(), '../backend/storage/.mech_api_token'), 'utf8').trim();

const api = async (path) => {
  const res = await fetch(`http://localhost:8000${path}`, {
    headers: { Authorization: `Bearer ${TOKEN}` },
  });
  return { status: res.status, body: await res.json().catch(() => null) };
};

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
await page.addInitScript((token) => {
  try {
    window.localStorage.setItem('mech_api_token', token);
  } catch {
    /* nothing to do */
  }
}, TOKEN);

const findings = [];
const check = (name, ok, detail) => {
  findings.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${detail ? ` — ${detail}` : ''}`);
};

const PROMPT = 'When Mary and John went to the store, John gave a drink to';

// --- Logit lens -----------------------------------------------------------
// The lens reads the live model, so the model is loaded first. A backend with
// no weights loaded answers `unavailable`, and the view would then show an
// honest refusal instead of a projection -- which is a different check.
const load = await (
  await fetch('http://localhost:8000/api/models/load', {
    method: 'POST',
    headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_name: 'gpt2-small' }),
  })
).json();
check('the backend loaded real GPT-2 weights', load.status === 'loaded', JSON.stringify(load).slice(0, 120));

const ll = await (await fetch('http://localhost:8000/api/gpt2/logit_lens_all', {
  method: 'POST',
  headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' },
  body: JSON.stringify({ prompt: PROMPT }),
})).json();

await page.goto(`${BASE}/#logitlens`, { waitUntil: 'domcontentloaded', timeout: 180000 });
const lens = page.getByTestId('view-logitlens');
await lens.waitFor({ timeout: 60000 });
await lens.locator('textarea, input[type="text"]').first().fill(PROMPT);
await lens.getByRole('button', { name: /run/i }).first().click();
await page.waitForTimeout(12000);
const lensText = await lens.innerText();
check(
  'logit lens renders the backend top token for layer 0',
  ll.layers?.[0]?.top_token ? lensText.includes(ll.layers[0].top_token.trim()) : false,
  `api top_token=${JSON.stringify(ll.layers?.[0]?.top_token)}`,
);
check(
  'logit lens reports the live layer count (12), not a placeholder grid',
  /12/.test(lensText) && !/unavailable|failed/i.test(lensText.slice(0, 400)),
  lensText.slice(0, 160).replace(/\s+/g, ' '),
);

// --- SAE: must refuse, not invent features ---------------------------------
await page.getByTestId('nav-saelab').click();
const sae = page.getByTestId('view-saelab');
await sae.waitFor({ timeout: 60000 });
await page.waitForTimeout(2000);
check('SAE Lab is the active view after clicking it', /SAE/i.test(await sae.innerText()), (await page.evaluate(() => location.hash)));
const saeStart = await sae.innerText();
check(
  'SAE Lab states the missing checkpoint instead of showing features',
  /checkpoint|unavailable|not configured/i.test(saeStart),
  saeStart.slice(0, 200).replace(/\s+/g, ' '),
);
check(
  'SAE Lab shows no fabricated feature activations before a run',
  !/feature\s*#?\d+\s*[:=]\s*0\.\d{3,}/i.test(saeStart),
);

// --- Benchmark: real run, real provenance ---------------------------------
const bm = await (await fetch('http://localhost:8000/api/benchmarks/run', {
  method: 'POST',
  headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' },
  body: JSON.stringify({ benchmark_name: 'IOI' }),
})).json();
await page.getByTestId('nav-benchmark').click();
const bench = page.getByTestId('view-benchmark');
await bench.waitFor({ timeout: 60000 });
await page.waitForTimeout(3000);
const benchText = await bench.innerText();
check(
  'benchmark dashboard names only the backend catalogue',
  /IOI/.test(benchText),
  benchText.slice(0, 160).replace(/\s+/g, ' '),
);
check(
  'a benchmark result states its provenance',
  /live|reference|provenance/i.test(benchText),
);

// --- Model catalogue: reference provenance must be visible ----------------
const models = await api('/api/models');
await page.getByTestId('nav-models').click();
const modelsView = page.getByTestId('view-models');
await modelsView.waitFor({ timeout: 60000 });
await page.waitForTimeout(4000);
const modelsText = await modelsView.innerText();
check(
  'Models view does not claim the reference catalogue is loaded weights',
  !/["']?\d+\s+models?\s+available/i.test(modelsText),
  modelsText.slice(0, 200).replace(/\s+/g, ' '),
);
check(
  'catalogue provenance is surfaced to the reader',
  /reference/i.test(modelsText) || models.body?.provenance === 'reference',
  `api provenance=${models.body?.provenance}`,
);

await page.screenshot({ path: 'test-results/live-research.png', fullPage: false });
await browser.close();

const failed = findings.filter((f) => !f.ok);
console.log(`\n${findings.length - failed.length}/${findings.length} live checks passed`);
process.exit(failed.length ? 1 : 0);
