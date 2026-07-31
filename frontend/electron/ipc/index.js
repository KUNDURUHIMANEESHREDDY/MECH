'use strict';

const { registerSettingsHandlers, defaultSettings } = require('./settings');
const { registerRuntimeHandlers } = require('./runtime');
const { registerWorkspaceHandlers } = require('./workspace');
const { registerStorageHandlers } = require('./storage');
const { registerEventHandlers } = require('./events');

function registerIpcHandlers({ ipcMain, storage, pythonBridge, logger }) {
  registerSettingsHandlers({ ipcMain, storage, logger });
  registerRuntimeHandlers({ ipcMain, pythonBridge, logger });
  registerWorkspaceHandlers({ ipcMain, storage, logger });
  registerStorageHandlers({ ipcMain, storage, logger });
  registerEventHandlers({ ipcMain, logger });
}

module.exports = {
  registerIpcHandlers,
  defaultSettings
};
