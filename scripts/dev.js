'use strict';

/**
 * Wait for the Vite dev server to be reachable, then launch Electron
 * pointing at it. Keeps the dev experience simple: `npm run dev` in one
 * terminal launches both processes via concurrently.
 */

const { spawn } = require('node:child_process');
const http = require('node:http');
const path = require('node:path');

const URL = process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173';
const ROOT = path.join(__dirname, '..');

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

async function main() {
  try {
    await waitForServer(URL, 30_000);
  } catch (err) {
    console.error('[dev] Vite dev server did not become ready:', err.message);
    process.exit(1);
  }

  const electronBin = require('electron');
  const child = spawn(electronBin, ['.'], {
    cwd: ROOT,
    stdio: 'inherit',
    env: { ...process.env, VITE_DEV_SERVER_URL: URL }
  });
  child.on('close', (code) => process.exit(code ?? 0));
}

main();
