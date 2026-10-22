/**
 * Load the model once for the whole suite.
 *
 * Every spec that needs GPT-2 asks the UI to load it, and the backend
 * serialises those loads behind one model lock. With a cold model, the first
 * spec paid the load and every other spec queued behind it, so the specs that
 * waited longest were the ones that failed -- the product looked broken when
 * it was merely queued.
 *
 * Loading the weights here, once, before any spec starts, means the specs
 * find the model warm: a load of an already-loaded model is a second, not a
 * minute. The specs still exercise the UI's own load control; they just do
 * not all pay for the first load.
 */
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');

const API = process.env.MECH_E2E_API || 'http://localhost:8000';
const TOKEN_FILE = resolve(process.cwd(), '../backend/storage/.mech_api_token');
const MODEL = process.env.MECH_E2E_MODEL || 'gpt2-small';

module.exports = async function globalSetup() {
  let token = '';
  try {
    token = readFileSync(TOKEN_FILE, 'utf8').trim();
  } catch {
    token = '';
  }
  if (!token) {
    console.warn('[global-setup] no API token found; the specs will run against an unauthenticated backend');
    return;
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 300_000);
  try {
    const response = await fetch(`${API}/api/models/load`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ model_name: MODEL }),
      signal: controller.signal,
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      console.warn(`[global-setup] the model could not be loaded (HTTP ${response.status}): ${JSON.stringify(body).slice(0, 160)}`);
      return;
    }
    console.log(`[global-setup] ${MODEL} loaded: ${body.status} (${body.device ?? 'unknown device'})`);
  } catch (error) {
    console.warn(`[global-setup] the model could not be loaded: ${String(error).slice(0, 160)}`);
  } finally {
    clearTimeout(timer);
  }
};
