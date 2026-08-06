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
  // Use shell mode on Windows for .cmd/.bat shims. This works from PowerShell,
  // cmd.exe, and Git Bash/MSYS without nested-quote issues.
  const isWinCmd = process.platform === 'win32' && /\.(cmd|bat)$/i.test(cmd);
  const res = spawnSync(cmd, args, {
    stdio: 'inherit',
    cwd: ROOT,
    shell: isWinCmd,
    ...(opts || {}),
  });
  if (res.status !== 0) {
    console.error(`[build] ${cmd} ${args.join(' ')} failed with code ${res.status}`);
    process.exit(res.status || 1);
  }
}

function ensureRendererBuilt() {
  // Always rebuild: a stale dist/ silently ships an old UI (electron-builder
  // packages whatever is on disk, so skipping the Vite build hides changes).
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
