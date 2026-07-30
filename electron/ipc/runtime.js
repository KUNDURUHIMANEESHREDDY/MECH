'use strict';

function registerRuntimeHandlers({ ipcMain, pythonBridge, logger }) {
  ipcMain.handle('python:ping', async () => {
    if (!pythonBridge) return { ok: false, error: 'no_bridge' };
    try {
      const data = await pythonBridge.call('ping', {});
      return { ok: true, data };
    } catch (err) {
      return { ok: false, error: err.message };
    }
  });

  ipcMain.handle('python:call', async (_event, { method, payload }) => {
    if (!pythonBridge) throw new Error('Python bridge not available');
    return pythonBridge.call(method, payload || {});
  });

  ipcMain.handle('runtime:status', async () => {
    if (!pythonBridge) return { status: 'disconnected' };
    try {
      return await pythonBridge.call('runtime:status', {});
    } catch (err) {
      logger.error('runtime_status_error', { error: err.message });
      return { status: 'error', error: err.message };
    }
  });

  ipcMain.handle('runtime:analyzeTokens', async (_event, prompt) => {
    if (!pythonBridge) throw new Error('Python bridge unavailable');
    return pythonBridge.call('runtime:analyze_tokens', { prompt });
  });

  // ── GPT-2 live interpretability pipeline ──────────────────────────────────

  ipcMain.handle('gpt2:load', async (_event, payload) => {
    if (!pythonBridge) return { status: 'error', error: 'no_bridge' };
    try { return await pythonBridge.call('gpt2/load', payload || {}); }
    catch (err) { return { status: 'error', error: err.message }; }
  });

  ipcMain.handle('gpt2:runPrompt', async (_event, payload) => {
    if (!pythonBridge) return { status: 'error', error: 'no_bridge' };
    try { return await pythonBridge.call('gpt2/run_prompt', payload || {}); }
    catch (err) { return { status: 'error', error: err.message }; }
  });

  ipcMain.handle('gpt2:activations', async (_event, payload) => {
    if (!pythonBridge) return { status: 'error', error: 'no_bridge' };
    try { return await pythonBridge.call('gpt2/activations', payload || {}); }
    catch (err) { return { status: 'error', error: err.message }; }
  });

  ipcMain.handle('gpt2:attentionHead', async (_event, payload) => {
    if (!pythonBridge) return { status: 'error', error: 'no_bridge' };
    try { return await pythonBridge.call('gpt2/attention_head', payload || {}); }
    catch (err) { return { status: 'error', error: err.message }; }
  });

  ipcMain.handle('gpt2:patchHead', async (_event, payload) => {
    if (!pythonBridge) return { status: 'error', error: 'no_bridge' };
    try { return await pythonBridge.call('gpt2/patch_head', payload || {}); }
    catch (err) { return { status: 'error', error: err.message }; }
  });

  ipcMain.handle('gpt2:ioi', async (_event, payload) => {
    if (!pythonBridge) return { status: 'error', error: 'no_bridge' };
    try { return await pythonBridge.call('gpt2/ioi', payload || {}); }
    catch (err) { return { status: 'error', error: err.message }; }
  });
}

module.exports = { registerRuntimeHandlers };
