'use strict';

/**
 * Build the renderer with Vite, then package the desktop app with
 * electron-builder. Intended to be invoked from `npm run build`.
 */

const { spawnSync } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');

const ROOT = path.join(__dirname, '..');

function run(cmd, args, opts) {
  // eslint-disable-next-line no-console
  console.log(`[build] ${cmd} ${args.join(' ')}`);
  // spawnSync of .cmd/.bat files fails with EINVAL on Windows (Node bug,
  // e.g. nodejs/node#29843) — route them through cmd.exe instead.
  const isWinCmd = process.platform === 'win32' && /\.(cmd|bat)$/i.test(cmd);
  const res = isWinCmd
    ? spawnSync(
        'cmd.exe',
        ['/d', '/s', '/c', `"${cmd}" ${args.map(a => (/\s/.test(a) ? `"${a}"` : a)).join(' ')}`],
        { stdio: 'inherit', cwd: ROOT, ...(opts || {}) },
      )
    : spawnSync(cmd, args, { stdio: 'inherit', cwd: ROOT, ...(opts || {}) });
  if (res.status !== 0) {
    console.error(`[build] ${cmd} ${args.join(' ')} failed with code ${res.status}`);
    process.exit(res.status || 1);
  }
}

function ensureRendererBuilt() {
  const distIndex = path.join(ROOT, 'dist', 'index.html');
  if (fs.existsSync(distIndex)) return;
  const npmCmd = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  run(npmCmd, ['run', 'build:renderer']);
}

function main() {
  ensureRendererBuilt();
  // electron-builder looks at package.json's `build` block
  const builder = require('electron-builder');
  builder.build().catch(err => {
    console.error('[build] electron-builder failed:', err);
    process.exit(1);
  });
}

main();
