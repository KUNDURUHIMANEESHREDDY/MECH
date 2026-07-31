'use strict';

// The Python backend is an HTTP FastAPI server on :8000 (prefix /api/v1).
// All handlers proxy to it; the old JSON-lines stdio sidecar no longer exists.

const BACKEND = 'http://localhost:8000/api/v1';

async function callBackend(method, payload = {}) {
  const res = await fetch(`${BACKEND}/${method}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ method, payload }),
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
