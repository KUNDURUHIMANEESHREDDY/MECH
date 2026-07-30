import '@testing-library/jest-dom/vitest';

// Stub the preload-injected API for component tests. The real
// implementation lives in electron/preload.js and is wired up at runtime
// in Electron; in the browser-style test environment we expose a small
// mock here so components that call window.appApi don't crash.
if (typeof window !== 'undefined' && !window.appApi) {
  const makeDeferred = () => {
    let resolve, reject;
    const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
    return { promise, resolve, reject };
  };

  const noop = () => Promise.resolve();

  window.appApi = {
    getSettings: () => Promise.resolve({
      theme: 'system',
      gpu: { enabled: true, acceleration: 'auto' },
      cache: { enabled: true, maxSizeMb: 1024, location: '' },
      paths: { python: 'python', workspace: '', projects: '' }
    }),
    setSettings: (patch) => Promise.resolve(patch),
    resetSettings: () => Promise.resolve({}),
    listProjects: () => Promise.resolve([]),
    addProject: (p) => Promise.resolve(p),
    removeProject: () => Promise.resolve({ ok: true }),
    listRecentFiles: () => Promise.resolve([]),
    addRecentFile: (f) => Promise.resolve(f),
    clearRecentFiles: () => Promise.resolve({ ok: true }),
    pythonPing: () => Promise.resolve({ ok: true, data: { ok: true, echo: 'pong' } }),
    pythonCall: (method, payload) => Promise.resolve({ method, payload, echo: true }),
    startBuild: () => Promise.resolve({ ok: true, pid: 1, target: 'renderer' }),
    getBuildLogs: () => Promise.resolve([]),
    clearBuildLogs: () => Promise.resolve({ ok: true }),
    onBuildEvent: () => () => undefined,
    getAppLogs: () => Promise.resolve([]),
    openExternal: () => Promise.resolve({ ok: true }),
    showInFolder: () => Promise.resolve({ ok: true })
  };
}
