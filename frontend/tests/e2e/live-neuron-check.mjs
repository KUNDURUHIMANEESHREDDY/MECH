/**
 * Neuron inspection live check (one-off, not part of the suite).
 *
 * Opens Model Explorer against a real backend, loads GPT-2 through the UI, runs
 * a prompt, clicks a returned neuron, and then compares ONLY the neuron detail
 * panel against the payload `POST /api/gpt2/neuron` returned for that exact
 * neuron. The chip list itself is not asserted: chips legitimately come from
 * several different responses (the prompt run's active neurons, the layer's
 * neuron page), so counting them would test the harness, not the app.
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium } from 'playwright';

const BASE = process.env.MECH_E2E_BASE || 'http://localhost:5273';
const TOKEN =
  process.env.MECH_E2E_TOKEN ||
  readFileSync(resolve(process.cwd(), '../backend/storage/.mech_api_token'), 'utf8').trim();
const PROMPT = 'When Mary and John went to the store, John gave a drink to';

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

// Capture the neuron-detail payload the view itself requested.
const neuronPayloads = [];
page.on('response', async (res) => {
  if (!res.url().includes('/api/gpt2/neuron')) return;
  try {
    neuronPayloads.push({ url: res.url(), body: await res.json() });
  } catch {
    /* non-JSON body: reported by the failed-request bucket instead */
  }
});

await page.goto(`${BASE}/#explorer`, { waitUntil: 'domcontentloaded', timeout: 180000 });
const view = page.getByTestId('view-explorer');
await view.waitFor({ timeout: 60000 });

if (!(await view.getByTestId('model-loaded').count())) {
  await view.locator('#model-explorer-model').selectOption('gpt2-small');
  await view.getByTestId('model-load').click();
  await view.getByTestId('model-loaded').waitFor({ timeout: 180000 });
}

await view.locator('#model-explorer-prompt').fill(PROMPT);
await view.getByTestId('model-run-prompt').click();
await view.getByTestId('model-prompt-result').waitFor({ timeout: 180000 });
await page.waitForTimeout(2000);

const chips = view.locator('.neuron-chip');
const chipCount = await chips.count();
if (!chipCount) {
  console.log('FAIL no neuron chips rendered after a live prompt run');
  await browser.close();
  process.exit(1);
}
const chipText = (await chips.first().innerText()).replace(/\s+/g, ' ');
// The detail panel is filled by the response to the neuron request. Wait for
// that response, then for the panel to name the neuron it returned: the
// panel's own empty state is non-empty text, so "the panel has text" is not
// evidence that the detail arrived.
const neuronResponse = page
  .waitForResponse(
    (res) => res.url().includes('/api/gpt2/neuron') && res.request().method() === 'POST',
    { timeout: 60_000 },
  )
  .catch(() => null);
await chips.first().click();
const neuronBody = await neuronResponse?.json?.().catch(() => null) ?? null;
const neuronId = typeof neuronBody?.id === 'string' ? neuronBody.id : null;
if (neuronId) {
  await page
    .waitForFunction(
      (want) => (document.querySelector('#component-panel-neuron')?.textContent ?? '').includes(want),
      neuronId,
      { timeout: 60_000 },
    )
    .catch(() => undefined);
}
await page.waitForTimeout(1000);

const detail = await view.locator('#component-panel-neuron').innerText();
const payload = neuronPayloads.at(-1)?.body ?? null;

const shown = {
  bias: payload?.bias,
  id: payload?.id,
  path: payload?.path,
  dMlp: payload?.d_mlp,
};
const formatted = (v) => (typeof v === 'number' ? v.toPrecision(4) : null);
const matches = {
  biasShown: shown.bias !== undefined && formatted(shown.bias) ? detail.includes(formatted(shown.bias)) : null,
  idShown: shown.id ? detail.includes(String(shown.id)) : null,
  pathShown: shown.path ? detail.includes(String(shown.path)) : null,
  dMlpShown: shown.dMlp ? detail.includes(String(shown.dMlp)) : null,
};

console.log('chip clicked:', chipText);
console.log('neuron payloads captured:', neuronPayloads.length);
console.log('payload field values:', shown);
console.log('rendered in detail panel:', matches);
console.log('detail panel text:', detail.replace(/\s+/g, ' ').slice(0, 400));

await page.screenshot({ path: 'test-results/live-neurons.png' });
await browser.close();

const verified = Object.values(matches).some((v) => v === true);
console.log(
  verified
    ? 'PASS detail panel shows backend-returned neuron values'
    : 'FAIL no backend neuron value visible in the detail panel',
);
process.exit(verified ? 0 : 1);
