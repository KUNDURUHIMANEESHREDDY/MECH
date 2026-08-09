'use strict';

// The Python backend runs as an in-process stdio sidecar (backend/mech_service.py).
// All handlers proxy through the PythonBridge ('http' method), which executes the
// FastAPI app via TestClient — no HTTP server, no ports.

let getApiKey = () => '';
let pythonBridge = null;

function setApiKeyResolver(resolver) {
  getApiKey = resolver || (() => '');
}

async function callBackend(method, payload = {}) {
  if (!pythonBridge) throw new Error('python_bridge_not_ready');
  const body = JSON.stringify({ method, payload });
  const res = await pythonBridge.call('http', {
    method: 'POST',
    path: `/api/v1/${method}`,
    headers: {
      'Content-Type': 'application/json',
      ...(getApiKey() ? { 'X-API-Key': getApiKey() } : {}),
    },
    body,
  });
  const status = typeof res.status === 'number' ? res.status : 200;
  if (status >= 400) throw new Error(`backend ${method} -> HTTP ${status}`);
  const json = JSON.parse(res.body || '{}');
  return json.result !== undefined ? json.result : json;
}

function registerRuntimeHandlers({ ipcMain, pythonBridge: bridge, logger, getApiKey }) {
  pythonBridge = bridge;
  setApiKeyResolver(getApiKey);
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
