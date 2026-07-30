/**
 * Browser Polyfill for window.appApi & window.electronAPI
 * Runs immediately on module import to bridge web frontend with Python Backend HTTP API Server on port 8000.
 */

if (typeof window !== 'undefined') {
  const BACKEND_URL = 'http://localhost:8000';

  const makeBackendCall = async (method, payload = {}) => {
    try {
      const res = await fetch(`${BACKEND_URL}/${method}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
