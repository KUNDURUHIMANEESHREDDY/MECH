'use strict';

const path = require('node:path');

function registerWorkspaceHandlers({ ipcMain, storage, logger }) {
  // ------- Projects / Workspaces -------
  ipcMain.handle('projects:list', async () => storage.list('projects'));
  ipcMain.handle('projects:add', async (_event, project) => {
    if (!project || !project.id) throw new Error('Project requires id');
    const now = new Date().toISOString();
    const record = {
      id: project.id,
      name: project.name || 'Untitled Workspace',
      path: project.path || '',
      createdAt: project.createdAt || now,
      updatedAt: now
    };
    await storage.upsert('projects', record);
    logger.info('project_added', { id: record.id });
    return record;
  });
  ipcMain.handle('projects:remove', async (_event, id) => {
    await storage.remove('projects', { id });
    logger.info('project_removed', { id });
    return { ok: true };
  });

  // ------- Recent Files / Recent Sessions -------
  ipcMain.handle('recent:list', async () => storage.list('recent_files'));
  ipcMain.handle('recent:add', async (_event, file) => {
    if (!file || !file.path) throw new Error('Recent file requires path');
    const record = {
      id: `rec_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      path: file.path,
      label: file.label || path.basename(file.path),
      openedAt: new Date().toISOString()
    };
    await storage.upsert('recent_files', record);
    return record;
  });
  ipcMain.handle('recent:clear', async () => {
    await storage.clear('recent_files');
    return { ok: true };
  });

  // ------- Sessions (Neural Debugger domain) -------
  ipcMain.handle('sessions:list', async () => storage.list('sessions'));
  ipcMain.handle('sessions:upsert', async (_event, session) => {
    if (!session || !session.id) throw new Error('Session requires id');
    const record = await storage.upsert('sessions', session);
    logger.info('session_saved', { id: record.id });
    return record;
  });
  ipcMain.handle('sessions:remove', async (_event, id) => {
    await storage.remove('sessions', { id });
    return { ok: true };
  });

  // ------- Experiments (Neural Debugger domain) -------
  ipcMain.handle('experiments:list', async () => storage.list('experiments'));
  ipcMain.handle('experiments:upsert', async (_event, exp) => {
    if (!exp || !exp.id) throw new Error('Experiment requires id');
    const record = await storage.upsert('experiments', exp);
    logger.info('experiment_saved', { id: record.id });
    return record;
  });
  ipcMain.handle('experiments:remove', async (_event, id) => {
    await storage.remove('experiments', { id });
    return { ok: true };
  });

  // ------- Layout Persistence & Bundle Export/Import -------
  ipcMain.handle('workspace:export_bundle', async () => {
    const projects = await storage.list('projects');
    const sessions = await storage.list('sessions');
    const experiments = await storage.list('experiments');
    return {
      version: '2.0.0',
      exported_at: new Date().toISOString(),
      projects,
      sessions,
      experiments,
    };
  });

  ipcMain.handle('workspace:import_bundle', async (_event, bundle) => {
    if (!bundle) throw new Error('Bundle payload is required');
    if (bundle.projects) {
      for (const p of bundle.projects) await storage.upsert('projects', p);
    }
    if (bundle.sessions) {
      for (const s of bundle.sessions) await storage.upsert('sessions', s);
    }
    if (bundle.experiments) {
      for (const e of bundle.experiments) await storage.upsert('experiments', e);
    }
    return { ok: true, imported_at: new Date().toISOString() };
  });
}

module.exports = { registerWorkspaceHandlers };
