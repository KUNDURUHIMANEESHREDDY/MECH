'use strict';

function defaultSettings() {
  return {
    theme: 'system', // 'light' | 'dark' | 'system'
    gpu: {
      enabled: true,
      acceleration: 'auto' // 'auto' | 'on' | 'off'
    },
    cache: {
      enabled: true,
      maxSizeMb: 1024,
      location: ''
    },
    paths: {
      python: process.platform === 'win32' ? 'python' : 'python3',
      workspace: '',
      projects: ''
    },
    models: {
      defaultModel: 'GPT-2 Small',
      device: 'cpu',
      maxContextTokens: 2048,
      precision: 'float32'
    },
    python: {
      path: process.platform === 'win32' ? 'python' : 'python3',
      environment: 'system',
      venvPath: ''
    },
    performance: {
      threads: 4,
      memoryLimitGb: 8,
      enableTorchCompile: false
    },
    plugins: {
      enabledPlugins: ['circuit-vis', 'token-heatmaps'],
      autoUpdate: true
    },
    debugger: {
      breakpointBehavior: 'pause',
      tokenHighlighting: 'activation',
      traceLevel: 'medium'
    },
    logging: {
      level: 'info',
      fileRetentionDays: 7,
      consoleOutput: true
    }
  };
}

function registerSettingsHandlers({ ipcMain, storage, logger }) {
  ipcMain.handle('settings:get', async () => {
    const current = await storage.get('settings');
    if (!current) return defaultSettings();
    return mergeDeep(defaultSettings(), current);
  });

  ipcMain.handle('settings:set', async (_event, patch) => {
    const current = (await storage.get('settings')) || defaultSettings();
    const next = mergeDeep(current, patch || {});
    await storage.set('settings', next);
    logger.info('settings_updated', { keys: Object.keys(patch || {}) });
    return next;
  });

  ipcMain.handle('settings:reset', async () => {
    const fresh = defaultSettings();
    await storage.set('settings', fresh);
    logger.info('settings_reset');
    return fresh;
  });
}

function mergeDeep(target, source) {
  if (Array.isArray(source)) return source.slice();
  if (source && typeof source === 'object') {
    const out = { ...target };
    for (const [k, v] of Object.entries(source)) {
      out[k] = v && typeof v === 'object' && !Array.isArray(v) ? mergeDeep(target?.[k] || {}, v) : v;
    }
    return out;
  }
  return source;
}

module.exports = { registerSettingsHandlers, defaultSettings };
