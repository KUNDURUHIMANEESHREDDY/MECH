'use strict';

/**
 * Build the renderer with Vite, then package the desktop app with
 * electron-builder. Intended to be invoked from `npm run build`.
 */

const { spawnSync } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');

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

/**
 * Check the extraResources declared in package.json's `build` block.
 *
 * `package.json` copies `../backend` and `../.venv` into the app bundle. A
 * missing `from` path is not reported clearly by electron-builder -- it fails
 * partway through packaging with an error that does not name the absent
 * directory, which is why `frontend/release/` had never been produced here with
 * no indication of why.
 *
 * Failing here names it.
 */
function checkExtraResources() {
  const pkg = require(path.join(ROOT, 'package.json'));
  const entries = (pkg.build && pkg.build.extraResources) || [];
  const missing = [];
  for (const entry of entries) {
    if (!entry || !entry.from) continue;
    const resolved = path.resolve(ROOT, entry.from);
    if (!fs.existsSync(resolved)) missing.push({ from: entry.from, resolved });
  }
  if (missing.length) {
    console.error('[build] package.json declares extraResources that do not exist:');
    for (const m of missing) console.error(`           ${m.from}  ->  ${m.resolved}`);
    console.error('[build] The app bundle needs the backend and a Python virtualenv.');
    console.error('[build] Create one with:  python -m venv .venv');
    console.error('[build] Or remove the entry from package.json `build.extraResources`.');
    process.exit(1);
  }
}

/**
 * Windows-specific: electron-builder unpacks a `winCodeSign` cache whose
 * contents include macOS `.dylib` symlinks. Creating a symlink on Windows needs
 * Developer Mode or an elevated shell; without it the extraction fails with
 * `Cannot create symbolic link : A required privilege is not held by the
 * client`, which surfaces only as `ERR_ELECTRON_BUILDER_CANNOT_EXECUTE` and
 * looks like a corrupt install.
 *
 * Detected up front so the cause is named instead of guessed at.
 */
function checkSymlinkPrivilege() {
  if (process.platform !== 'win32') return;
  const probe = path.join(os.tmpdir(), `mech-symlink-probe-${process.pid}`);
  try {
    fs.symlinkSync(probe, `${probe}-link`, 'file');
    fs.unlinkSync(`${probe}-link`);
  } catch (err) {
    console.error('[build] This Windows account cannot create symbolic links.');
    console.error(`[build] electron-builder needs them to unpack its winCodeSign cache.`);
    console.error(`[build]   ${err.message}`);
    console.error('[build] Enable Developer Mode (Settings > Update > For developers),');
    console.error('[build] or run this build from an elevated shell.');
    process.exit(1);
  }
}

function main() {
  checkExtraResources();
  checkSymlinkPrivilege();
  ensureRendererBuilt();
  // electron-builder looks at package.json's `build` block
  const builder = require('electron-builder');
  builder.build().catch(err => {
    console.error('[build] electron-builder failed:', err);
    process.exit(1);
  });
}

main();
