'use strict';

const { spawn } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');

function broadcast(buildEvents, channel, payload) {
  for (const wc of buildEvents.senders) {
    try {
      wc.send(channel, payload);
    } catch {
      // sender may be gone; ignore
    }
  }
}

function registerEventHandlers({ ipcMain, logger }) {
  const buildState = {
    running: false,
    process: null,
    logBuffer: [],
    senders: new Set()
  };

  function appendLog(level, message, meta) {
    const entry = {
      ts: new Date().toISOString(),
      level,
      message,
      meta: meta || null
    };
    buildState.logBuffer.push(entry);
    if (buildState.logBuffer.length > 1000) buildState.logBuffer.shift();
    logger.info(`build_${level}`, { message, ...(meta || {}) });
  }

  ipcMain.handle('build:start', async (event, options) => {
    if (buildState.running) {
      return { ok: false, error: 'build_already_running' };
    }
    const target = (options && options.target) || 'renderer';
    const cmd = target === 'python' ? resolvePythonCommand() : resolveRendererCommand();

    buildState.senders.add(event.sender);
    buildState.running = true;
    appendLog('info', `build_started target=${target}`);

    const child = spawn(cmd.command, cmd.args, {
      cwd: path.join(__dirname, '..', '..'),
      env: { ...process.env, FORCE_COLOR: '0' },
      shell: false
    });
    buildState.process = child;

    const pushEvent = (type, data) => {
      const payload = { type, ts: new Date().toISOString(), data };
      broadcast(buildState, 'build:event', payload);
    };

    child.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      appendLog('info', text.trimEnd());
      pushEvent('stdout', text);
    });
    child.stderr.on('data', (chunk) => {
      const text = chunk.toString();
      appendLog('warn', text.trimEnd());
      pushEvent('stderr', text);
    });
    child.on('error', (err) => {
      appendLog('error', err.message);
      pushEvent('error', err.message);
    });
    child.on('close', (code) => {
      buildState.running = false;
      buildState.process = null;
      const level = code === 0 ? 'info' : 'error';
      appendLog(level, `build_finished code=${code}`);
      pushEvent('close', { code });
      buildState.senders.delete(event.sender);
    });

    return { ok: true, pid: child.pid, target };
  });

  ipcMain.handle('build:logs', async () => buildState.logBuffer.slice());
  ipcMain.handle('build:clear', async () => {
    buildState.logBuffer = [];
    return { ok: true };
  });

  ipcMain.handle('logs:get', async () => logger.getEntries());

  ipcMain.handle('system:openExternal', async (_event, url) => {
    const { shell } = require('electron');
    // Only allow http(s) links to leave the app. Reject file:, javascript:,
    // data: and any other scheme that shell.openExternal would otherwise honour.
    let safe = false;
    try {
      const parsed = new URL(url);
      safe = parsed.protocol === 'http:' || parsed.protocol === 'https:';
    } catch {
      safe = false;
    }
    if (safe) {
      await shell.openExternal(url);
    }
    return { ok: safe };
  });
  ipcMain.handle('system:showInFolder', async (_event, p) => {
    const { shell } = require('electron');
    if (p && fs.existsSync(p)) shell.showItemInFolder(p);
    return { ok: true };
  });
}

function resolvePythonCommand() {
  if (process.platform === 'win32') {
    return { command: 'python', args: ['-m', 'pip', '--version'] };
  }
  return { command: 'python3', args: ['-m', 'pip', '--version'] };
}

function resolveRendererCommand() {
  return { command: process.platform === 'win32' ? 'npm.cmd' : 'npm', args: ['run', 'build:renderer'] };
}

module.exports = { registerEventHandlers };
