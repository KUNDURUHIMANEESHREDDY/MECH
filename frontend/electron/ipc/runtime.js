'use strict';

// The Python backend is an HTTP FastAPI server on :8000. All handlers proxy to
// it; the old JSON-lines stdio sidecar no longer exists.
//
// The prefix is /api. It was /api/v1, which the backend never served: main.py
// mounts the dispatcher at /api (and an optional v2 at /api/v2), so every
// proxied call 404'd. electron/main.js uses 127.0.0.1 rather than localhost to
// avoid IPv6 name resolution failing on Windows.

const BACKEND = 'http://127.0.0.1:8000/api';

async function callBackend(method, payload = {}) {
  // The body must be the payload itself, not {method, payload}. The dispatcher
  // endpoints declare `payload: Dict[str, Any]`, which FastAPI binds as the
  // request body, so a wrapped envelope made payload.get("prompt") return ""
  // and every field silently fall back to its default. Endpoints that take no
  // body (ping, runtime/status) ignore it either way.
  const res = await fetch(`${BACKEND}/${method}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload ?? {}),
  });
  if (!res.ok) throw new Error(`backend ${method} -> HTTP ${res.status}`);
  const json = await res.json();
  return json.result !== undefined ? json.result : json;
}

function registerRuntimeHandlers({ ipcMain, logger }) {
  ipcMain.handle('python:ping', async () => {
    try {
      const data = await callBackend('ping', {});
      return { ok: true, data };
    } catch (err) {
      return { ok: false, error: err.message };
    }
  });

  ipcMain.handle('python:call', async (_event, { method, payload }) => {
    return callBackend(method, payload || {});
  });

  ipcMain.handle('runtime:status', async () => {
    try {
      return await callBackend('runtime/status', {});
    } catch (err) {
      logger.error('runtime_status_error', { error: err.message });
      return { status: 'error', error: err.message };
    }
  });

  ipcMain.handle('runtime:analyzeTokens', async (_event, prompt) => {
    return callBackend('runtime/analyze_tokens', { prompt: prompt || '' });
  });

  // ── GPT-2 live interpretability pipeline ──────────────────────────────────

  ipcMain.handle('gpt2:load', async () => {
    try { return await callBackend('gpt2/load', {}); }
    catch (err) { return { status: 'error', error: err.message }; }
  });

  ipcMain.handle('gpt2:runPrompt', async (_event, payload) => {
    try { return await callBackend('gpt2/run_prompt', payload || {}); }
    catch (err) { return { status: 'error', error: err.message }; }
  });

  ipcMain.handle('gpt2:activations', async (_event, payload) => {
    try { return await callBackend('gpt2/activations', payload || {}); }
    catch (err) { return { status: 'error', error: err.message }; }
  });

  ipcMain.handle('gpt2:attentionHead', async (_event, payload) => {
    try { return await callBackend('gpt2/attention_head', payload || {}); }
    catch (err) { return { status: 'error', error: err.message }; }
  });

  ipcMain.handle('gpt2:patchHead', async (_event, payload) => {
    try { return await callBackend('gpt2/patch_head', payload || {}); }
    catch (err) { return { status: 'error', error: err.message }; }
  });

  ipcMain.handle('gpt2:ioi', async (_event, payload) => {
    try { return await callBackend('gpt2/ioi', payload || {}); }
    catch (err) { return { status: 'error', error: err.message }; }
  });
}

module.exports = { registerRuntimeHandlers };
