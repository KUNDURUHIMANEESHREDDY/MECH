/**
 * SAE Lab live check (one-off, not part of the suite).
 *
 * Drives the SAE Lab against a real backend and compares the features on
 * screen with the payload `POST /api/sae/inspect` returned. The screen is
 * never the oracle: the expected indices and activations come from the API
 * response the view itself received, and a feature the API did not return is
 * a failure even when it is rendered.
 *
 * Which state is checked is decided by what the backend reports, not by a
 * flag here:
 *
 *   - a checkpoint can be reached (the server has one configured, or the
 *     checkpoint directory lists files the request can name): the view must
 *     show exactly the returned indices and activations;
 *   - no checkpoint can be reached: the view must say so and render no
 *     features at all. An empty feature list is not "no feature fires".
 *
 * Env: MECH_E2E_BASE (default http://localhost:5273),
 *      MECH_E2E_API  (default http://localhost:8000),
 *      MECH_E2E_TOKEN, MECH_E2E_PROMPT.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium } from 'playwright';

const BASE = process.env.MECH_E2E_BASE || 'http://localhost:5273';
const API = process.env.MECH_E2E_API || 'http://localhost:8000';
const PROMPT = process.env.MECH_E2E_PROMPT || 'The capital of France is';

function readToken() {
  if (process.env.MECH_E2E_TOKEN) return process.env.MECH_E2E_TOKEN.trim();
  const candidates = [
    resolve(process.cwd(), '../backend/storage/.mech_api_token'),
    resolve(process.cwd(), 'backend/storage/.mech_api_token'),
  ];
  for (const path of candidates) {
    try {
      const token = readFileSync(path, 'utf8').trim();
      if (token) return token;
    } catch {
      /* try the next candidate */
    }
  }
  throw new Error('No API token found. Set MECH_E2E_TOKEN or keep backend/storage/.mech_api_token.');
}

const TOKEN = readToken();

/** `GET /api/sae/status` straight from the backend, as the truth about it. */
async function apiStatus() {
  const response = await fetch(`${API}/api/sae/status`, {
    headers: { Authorization: `Bearer ${TOKEN}` },
  });
  if (!response.ok) {
    throw new Error(`GET /api/sae/status failed: HTTP ${response.status}`);
  }
  return response.json();
}

/** A bare file name inside the checkpoint directory -- not a path. */
function isBareFilename(value) {
  const name = String(value || '').trim();
  if (!name || name === '.' || name === '..') return false;
  if (name.startsWith('.')) return false;
  if (name.includes('/') || name.includes('\\') || name.includes(':')) return false;
  return true;
}

async function readRenderedFeatures(view) {
  // `sae-feature-<index>` only: the filter input is `sae-feature-search`.
  const buttons = view.getByTestId(/^sae-feature-\d+$/);
  const count = await buttons.count();
  const rendered = [];
  for (let index = 0; index < count; index += 1) {
    const button = buttons.nth(index);
    const testid = (await button.getAttribute('data-testid')) || '';
    const text = (await button.innerText()).replace(/\s+/g, ' ').trim();
    const parts = text.split(' ');
    rendered.push({
      index: Number(testid.replace('sae-feature-', '')),
      text,
      activation: parts.length > 1 ? parts[parts.length - 1] : null,
    });
  }
  return rendered;
}

function check(label, ok, detail) {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}${detail ? ` — ${detail}` : ''}`);
  return ok;
}

const status = await apiStatus();
console.log('backend status:', JSON.stringify({
  checkpoint_configured: status.checkpoint_configured,
  checkpoint_source: status.checkpoint_source,
  checkpoint_identifier: status.checkpoint_identifier ?? '(field absent)',
  checkpoint_dir: status.checkpoint_dir ?? '(field absent)',
  available_checkpoints: Array.isArray(status.available_checkpoints)
    ? `${status.available_checkpoints.length} file(s)`
    : '(field absent)',
}));

const available = Array.isArray(status.available_checkpoints) ? status.available_checkpoints : [];
const configuredIdentifier = typeof status.checkpoint_identifier === 'string'
  ? status.checkpoint_identifier.trim()
  : '';
const serverHasCheckpoint = status.checkpoint_configured === true;
// A checkpoint is reachable when the server has one, or when a name can be
// sent explicitly. `available_checkpoints` is only that -- it says which
// files exist, not that any of them can be encoded.
const canNameCheckpoint = available.length > 0;
const canUseCheckpoint = serverHasCheckpoint || canNameCheckpoint;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

const inspectPayloads = [];
page.on('response', async (res) => {
  if (!res.url().includes('/api/sae/inspect')) return;
  try {
    inspectPayloads.push(await res.json());
  } catch {
    /* a non-JSON body is reported by the missing-payload check instead */
  }
});

await page.goto(`${BASE}/#saelab`, { waitUntil: 'domcontentloaded', timeout: 180000 });
const view = page.getByTestId('view-saelab');
await view.waitFor({ timeout: 60000 });
await view.getByTestId('sae-status').waitFor({ timeout: 60000 });
await page.waitForTimeout(1000);

await view.locator('#sae-prompt').fill(PROMPT);

let passed = true;

if (canUseCheckpoint) {
  // 1. Point the request at a real checkpoint.
  const select = view.getByTestId('sae-checkpoint-select');
  const optionCount = await select.locator('option').count();
  const usableOptions = await select.locator('option').evaluateAll((nodes) =>
    nodes.map((node) => node.value).filter((value) => value !== ''));

  if (usableOptions.length) {
    const preferred = usableOptions.includes(configuredIdentifier)
      ? configuredIdentifier
      : usableOptions[0];
    await select.selectOption(preferred);
    console.log('selected checkpoint from the list:', preferred);
  } else if (isBareFilename(configuredIdentifier)) {
    await view.locator('#sae-identifier').fill(configuredIdentifier);
    console.log('typed the server checkpoint name:', configuredIdentifier);
  } else {
    // The server holds a checkpoint that cannot be named in a request (an
    // absolute path); the request then uses the server's own configuration.
    console.log('using the server checkpoint configuration as-is');
  }

  // 2. Inspect and wait for the backend's answer.
  await view.getByTestId('sae-inspect').click();
  await Promise.race([
    view.getByTestId('sae-features').waitFor({ timeout: 180000 }),
    view.getByTestId('sae-inspect-failure').waitFor({ timeout: 180000 }),
  ]);
  await page.waitForTimeout(500);

  const payload = inspectPayloads.at(-1);
  if (!check('the view received an inspect response', Boolean(payload), payload ? '' : 'no payload captured')) {
    passed = false;
  } else {
    const apiIndices = Array.isArray(payload.feature_indices) ? payload.feature_indices : [];
    const apiActivations = Array.isArray(payload.activations) ? payload.activations : [];
    const rendered = await readRenderedFeatures(view);

    console.log('api feature indices:', JSON.stringify(apiIndices));
    console.log('api activations:', JSON.stringify(apiActivations));

    passed = check(
      'the API returned features',
      apiIndices.length > 0,
      `${apiIndices.length} index(es)`,
    ) && passed;

    passed = check(
      'every API feature index is rendered',
      apiIndices.every((index) => rendered.some((entry) => entry.index === index)),
      `rendered ${rendered.length} of ${apiIndices.length}`,
    ) && passed;

    passed = check(
      'no feature is rendered that the API did not return',
      rendered.every((entry) => apiIndices.includes(entry.index)),
      `rendered: ${rendered.map((entry) => entry.index).join(', ') || 'none'}`,
    ) && passed;

    const activationMismatches = [];
    apiIndices.forEach((index, position) => {
      const entry = rendered.find((item) => item.index === index);
      const expected = typeof apiActivations[position] === 'number'
        ? apiActivations[position].toFixed(6)
        : null;
      if (!entry) {
        activationMismatches.push(`#${index} not rendered`);
        return;
      }
      if (expected !== null && entry.activation !== expected) {
        activationMismatches.push(`#${index} rendered ${entry.activation}, API returned ${expected}`);
      }
    });
    passed = check(
      'every rendered activation is the value the API returned',
      activationMismatches.length === 0,
      activationMismatches.join('; ') || 'all match',
    ) && passed;

    // The reconstruction number on screen is the number the API returned, and
    // the tensor it refers to is named: a fidelity figure without the tensor
    // it was measured on is a number about nothing.
    const reconstruction = await view.getByTestId('sae-reconstruction').innerText().catch(() => '');
    const apiError = payload.relative_l2_error;
    if (payload.reconstruction_measured === true && typeof apiError === 'number') {
      const renderedError = reconstruction.includes(String(apiError))
        || reconstruction.includes(apiError.toFixed(6))
        || reconstruction.includes(String(Number(apiError.toFixed(6))));
      passed = check(
        'the rendered reconstruction error is the value the API returned',
        renderedError,
        `api relative_l2_error=${apiError}`,
      ) && passed;
    } else {
      passed = check(
        'an unmeasured reconstruction is shown as unavailable, not as a number',
        reconstruction.includes('NOT AVAILABLE'),
        reconstruction.replace(/\s+/g, ' ').slice(0, 120),
      ) && passed;
    }
    const apiTensor = String(payload.tensor_source ?? '');
    passed = check(
      'the view names the tensor the reconstruction was measured on',
      apiTensor ? reconstruction.includes(apiTensor) : true,
      apiTensor || 'the API named no tensor',
    ) && passed;

    const evidence = await view.getByTestId('sae-evidence').innerText().catch(() => '');
    console.log('evidence level shown:', evidence);
  }
} else {
  // No checkpoint anywhere: the honest refusal is the expected result.
  const warning = view.getByTestId('sae-checkpoint-warning');
  const warningVisible = await warning.count();
  passed = check(
    'the view states that no checkpoint is configured',
    warningVisible > 0,
    warningVisible ? (await warning.innerText()).replace(/\s+/g, ' ').slice(0, 160) : 'no warning shown',
  ) && passed;

  const listing = await view.getByTestId('sae-checkpoint-listing').innerText().catch(() => '');
  console.log('checkpoint listing note:', listing.replace(/\s+/g, ' '));

  await view.getByTestId('sae-inspect').click();
  await Promise.race([
    view.getByTestId('sae-inspect-failure').waitFor({ timeout: 60000 }),
    view.getByTestId('sae-features').waitFor({ timeout: 60000 }),
  ]);
  await page.waitForTimeout(500);

  const failure = view.getByTestId('sae-inspect-failure');
  passed = check(
    'the backend refuses instead of returning features',
    (await failure.count()) > 0,
    (await failure.count()) ? (await failure.innerText()).replace(/\s+/g, ' ').slice(0, 160) : 'no failure shown',
  ) && passed;

  const rendered = await readRenderedFeatures(view);
  passed = check(
    'no feature is rendered',
    rendered.length === 0,
    `rendered ${rendered.length}`,
  ) && passed;
}

await page.screenshot({ path: 'test-results/live-sae.png' });
await browser.close();

console.log(passed
  ? 'PASS SAE Lab matches the API response'
  : 'FAIL SAE Lab does not match the API response');
process.exit(passed ? 0 : 1);
