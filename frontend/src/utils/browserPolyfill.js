/**
 * Browser Polyfill for window.appApi & window.electronAPI
 *
 * Runs on import to bridge the web build to the Python backend on port 8000.
 *
 * No credential lives here. The browser build authenticates with the HttpOnly
 * session cookie issued by `POST /session`: JavaScript cannot read that cookie,
 * and every request is sent with `credentials: 'include'` so the browser
 * attaches it. `include` (not `same-origin`) is deliberate: the dev server runs
 * on :5173 and talks to :8000, and the backend's CORS config allows credentials
 * for loopback origins. An earlier version resolved a bearer token from
 * `localStorage` and attached it as a header, which put the control-plane
 * credential in reach of any script on the page.
 */

if (typeof window !== 'undefined') {
  const BACKEND_URL = 'http://localhost:8000';

  const makeBackendCall = async (method, payload = {}) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/${method}`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        // The dispatcher binds `payload: Dict` as the request body, so the body
        // is the payload itself -- not a {method, payload} envelope.
        body: JSON.stringify(payload ?? {}),
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
