'use strict';

/**
 * One-command dev startup: spawns the Python backend (auto-detecting a
 * Python with torch installed), waits for it to be healthy, then waits
 * for the Vite dev server and launches Electron pointing at it.
 */

const { spawn, execFile } = require('node:child_process');
const http = require('node:http');
const net = require('node:net');
const path = require('node:path');

const DEV_URL = process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173';
const ROOT = path.join(__dirname, '..');
const PROJECT_ROOT = path.join(ROOT, '..');
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';
const BACKEND_PORT = new URL(BACKEND_URL).port;

function waitForServer(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const tryOnce = () => {
      const req = http.get(url, (res) => {
        res.resume();
        if (res.statusCode && res.statusCode < 500) return resolve();
        retry();
      });
      req.on('error', retry);
      req.setTimeout(1000, () => { req.destroy(); retry(); });
    };
    const retry = () => {
      if (Date.now() > deadline) return reject(new Error('vite_dev_timeout'));
      setTimeout(tryOnce, 400);
    };
    tryOnce();
  });
}

function isPortOpen(port) {
  return new Promise((resolve) => {
    const sock = net.connect({ port, host: '127.0.0.1' });
    sock.once('connect', () => { sock.destroy(); resolve(true); });
    sock.once('error', () => resolve(false));
  });
}

function hasTorch(cmd, args) {
  return new Promise((resolve) => {
    execFile(cmd, [...args, '-c', 'import torch, transformers'], { timeout: 60000 }, (err) => resolve(!err));
  });
}

async function findPython() {
  const candidates = [
    ['C:\\Program Files\\Python312\\python.exe', []],
    ['C:\\Program Files\\Python313\\python.exe', []],
    ['C:\\Python312\\python.exe', []],
    ['python', []],
    ['py', ['-3']],
  ];
  for (const [cmd, args] of candidates) {
    try {
      if (await hasTorch(cmd, args)) return { cmd, args };
    } catch (e) { /* candidate not found, try next */ }
  }
  return null;
}

async function ensureBackend() {
  if (await isPortOpen(BACKEND_PORT)) {
    console.log(`[dev] Backend already running on ${BACKEND_URL} — reusing it.`);
    return null;
  }
  const py = await findPython();
  if (!py) {
    console.warn('[dev] No Python with torch found on PATH. Start the backend manually (python backend/main.py).');
    return null;
  }
  console.log(`[dev] Spawning backend: ${py.cmd} backend/main.py`);
  const child = spawn(py.cmd, [...py.args, 'backend/main.py'], {
    cwd: PROJECT_ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONPATH: `${PROJECT_ROOT};${path.join(PROJECT_ROOT, 'backend')}` },
  });
  child.stdout.on('data', (d) => process.stdout.write(`[backend] ${d}`));
  child.stderr.on('data', (d) => process.stderr.write(`[backend] ${d}`));
  child.on('exit', (code) => console.log(`[dev] Backend exited with code ${code}`));

  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    if (await isPortOpen(BACKEND_PORT)) {
      console.log(`[dev] Backend healthy on ${BACKEND_URL}`);
      return child;
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  console.warn('[dev] Backend did not become healthy within 60s.');
  return child;
}

async function main() {
  let backend = await ensureBackend();

  const killBackend = () => { if (backend) { try { backend.kill(); } catch (e) { /* already gone */ } } };
  process.on('exit', killBackend);
  process.on('SIGINT', () => { killBackend(); process.exit(0); });
  process.on('SIGTERM', () => { killBackend(); process.exit(0); });

  try {
    await waitForServer(DEV_URL, 30_000);
  } catch (err) {
    console.error('[dev] Vite dev server did not become ready:', err.message);
    killBackend();
    process.exit(1);
  }

  const electronBin = require('electron');
  const child = spawn(electronBin, ['.'], {
    cwd: ROOT,
    stdio: 'inherit',
    env: { ...process.env, VITE_DEV_SERVER_URL: DEV_URL }
  });
  child.on('close', (code) => { killBackend(); process.exit(code ?? 0); });
}

main();
