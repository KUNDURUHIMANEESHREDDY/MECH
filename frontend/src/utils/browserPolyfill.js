/**
 * Browser Polyfill for window.appApi & window.electronAPI
 * Runs immediately on module import to bridge web frontend with Python Backend HTTP API Server on port 8000.
 */

if (typeof window !== 'undefined') {
  const BACKEND_URL = 'http://localhost:8000';

  const makeBackendCall = async (method, payload = {}) => {
    try {
      const apiKey = (await window.desktopApi?.getApiKey?.()) || '';
      const res = await fetch(`${BACKEND_URL}/api/v1/${method}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiKey ? { 'X-API-Key': apiKey } : {}),
        },
        body: JSON.stringify({ method, payload }),
      });
      const json = await res.json();
      return json.result !== undefined ? json.result : json;
    } catch (err) {
      console.warn(`[Backend API] Call failed for ${method}:`, err);
      return { error: err.message };
    }
  };

  if (!window.appApi) {
    window.appApi = {
      getSettings: async () => ({ theme: 'dark' }),
      setSettings: async (s) => s,
      pythonPing: async () => makeBackendCall('ping', {}),
      pythonCall: async (method, payload) => makeBackendCall(method, payload),
      getRuntimeStatus: async () => makeBackendCall('runtime/status', {}),
      analyzeTokens: async (prompt) => makeBackendCall('runtime/analyze_tokens', { prompt }),

      // GPT-2 live interpretability pipeline
      gpt2Load: async () => makeBackendCall('gpt2/load', {}),
      gpt2RunPrompt: async (prompt) => makeBackendCall('gpt2/run_prompt', { prompt }),
      gpt2GetActivations: async (layer) => makeBackendCall('gpt2/activations', { layer }),
      gpt2AttentionHead: async (layer, head) => makeBackendCall('gpt2/attention_head', { layer, head }),
      gpt2PatchHead: async (layer, head, posToken, negToken) =>
        makeBackendCall('gpt2/patch_head', { layer, head, pos_token: posToken, neg_token: negToken }),
      gpt2RunIoi: async (ioName, subjName) =>
        makeBackendCall('gpt2/ioi', { io_name: ioName, subj_name: subjName }),

      listProjects: async () => [],
      listRecentFiles: async () => [],
      listSessions: async () => [],
      listExperiments: async () => [],
    };
  }

  if (!window.electronAPI) {
    window.electronAPI = {
      invoke: async (method, payload) => makeBackendCall(method, payload),
    };
  }
}
