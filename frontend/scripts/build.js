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
 * Decide which declared extraResources actually exist.
 *
 * `package.json` copies `../backend` and `../.venv` into the app bundle. A
 * missing `from` path is not reported clearly by electron-builder -- it fails
 * partway through packaging with an error that does not name the absent
 * directory, which is why `frontend/release/` had never been produced here with
 * no indication of why.
 *
 * `../backend` is required: without it the app has no backend to start.
 * `../.venv` is not. `electron/main.js:resolvePythonPath` already falls back
 * through the bundled `.venv`, the repo `.venv`, the repo `venv`, and finally
 * bare `python` -- so a build without a bundled interpreter produces an app that
 * still runs, using whatever interpreter the machine has.
 *
 * Failing the whole build over a preference would mean no release artifact at
 * all, so the entry is dropped and the omission is reported rather than hidden.
 */
function resolveExtraResources() {
  const pkg = require(path.join(ROOT, 'package.json'));
  const entries = (pkg.build && pkg.build.extraResources) || [];
  const present = [];
  const dropped = [];

  for (const entry of entries) {
    if (!entry || !entry.from) {
      dropped.push({ from: '(malformed entry)', required: false });
      continue;
    }
    const resolved = path.resolve(ROOT, entry.from);
    if (fs.existsSync(resolved)) {
      present.push(entry);
      continue;
    }
    const required = /backend/i.test(entry.from);
    dropped.push({ from: entry.from, resolved, required });
  }

  const missingRequired = dropped.filter((d) => d.required);
  if (missingRequired.length) {
    console.error('[build] package.json declares a required extraResource that does not exist:');
    for (const m of missingRequired) {
      console.error(`           ${m.from}  ->  ${m.resolved}`);
    }
    console.error('[build] The app bundle cannot start a backend it does not contain.');
    process.exit(1);
  }

  for (const m of dropped) {
    console.log(`[build] skipping absent optional extraResource: ${m.from}`);
    if (/\.venv/i.test(m.from)) {
      console.log('[build]   the app will resolve an interpreter at runtime via');
      console.log('[build]   electron/main.js:resolvePythonPath (bundled .venv, repo');
      console.log('[build]   .venv, repo venv, then bare python).');
    }
  }

  return present;
}

/**
 * Windows-specific: electron-builder unpacks a `winCodeSign` cache whose
 * contents include macOS `.dylib` symlinks. Creating a symlink on Windows needs
 * Developer Mode or an elevated shell; without it the extraction fails with
 * `Cannot create symbolic link : A required privilege is not held by the
 * client`, which surfaces only as `ERR_ELECTRON_BUILDER_CANNOT_EXECUTE` and
 * looks like a corrupt install.
 *
 * That cache is only needed to re-sign and stamp the built executable. Skipping
 * `signAndEditExecutable` avoids needing it at all, which is what makes a local
 * unsigned build possible without Developer Mode. The tradeoff is real and is
 * reported: the resulting executable is neither signed nor version-stamped, so
 * SmartScreen will warn and the file has no embedded version metadata.
 */
function windowsSigningConfig() {
  if (process.platform !== 'win32') return {};
  const probe = path.join(os.tmpdir(), `mech-symlink-probe-${process.pid}`);
  try {
    fs.symlinkSync(probe, `${probe}-link`, 'file');
    fs.unlinkSync(`${probe}-link`);
    return {};
  } catch (err) {
    console.log('[build] This account cannot create symlinks, so electron-builder');
    console.log('[build] cannot unpack its winCodeSign cache:');
    console.log(`[build]   ${err.message}`);
    console.log('[build] Continuing with signAndEditExecutable disabled -- the');
    console.log('[build] artifact will be UNSIGNED and unversioned.');
    console.log('[build] Enable Developer Mode (Settings > Update > For developers)');
    console.log('[build] or build from an elevated shell for a signed artifact.');
    return { signAndEditExecutable: false };
  }
}

/**
 * Build the effective electron-builder config.
 *
 * The whole `build` block from package.json is reproduced here and then amended,
 * rather than passing an overlay to `builder.build()`. An overlay is not merged
 * by electron-builder -- it replaces, so `{ win: {...} }` arrives with no `files`,
 * no `extraResources` and no `directories`, and the run dies with
 * `TypeError: types is not iterable` from deep inside the packager.
 */
function effectiveConfig(extraResources) {
  const pkg = require(path.join(ROOT, 'package.json'));
  const base = pkg.build || {};

  const config = Object.assign({}, base, {
    extraResources,
    directories: Object.assign({}, base.directories, { output: 'release' }),
    files: base.files,
  });

  const winOverrides = windowsSigningConfig();
  if (Object.keys(winOverrides).length) {
    config.win = Object.assign({}, base.win, winOverrides);
  }
  return config;
}

function main() {
  const extraResources = resolveExtraResources();
  ensureRendererBuilt();
  const builder = require('electron-builder');
  builder.build({ config: effectiveConfig(extraResources) })
    .then((paths) => {
      console.log('[build] done:', Array.isArray(paths) ? paths.join(', ') : paths);
    })
    .catch(err => {
      console.error('[build] electron-builder failed:', err);
      process.exit(1);
    });
}

main();
