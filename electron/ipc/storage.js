'use strict';

function registerStorageHandlers({ ipcMain, storage, logger }) {
  ipcMain.handle('storage:list', async (_event, table) => storage.list(table));
  ipcMain.handle('storage:upsert', async (_event, { table, record }) => storage.upsert(table, record));
  ipcMain.handle('storage:remove', async (_event, { table, where }) => storage.remove(table, where));
  ipcMain.handle('storage:clear', async (_event, table) => storage.clear(table));
}

module.exports = { registerStorageHandlers };
