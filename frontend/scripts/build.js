'use strict';

/**
 * Build the renderer with Vite, then package the desktop app with
 * electron-builder. Intended to be invoked from `npm run build`.
 */

const { spawnSync } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');

const { stageBackend } = require('./backend_source_set');

const ROOT = path.join(__dirname, '..');

function run(cmd, args, opts) {
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
 * Stage the backend into `build/backend` before the packaging config is
 * resolved.
 *
 * The packaged backend must be exactly the repository's source for
 * `backend/`: the tracked files plus the untracked-but-not-ignored ones,
 * minus the secrets and runtime state. The electron-builder filter is the
 * second line of defence; this staging is the first.
 *
 * A failure here stops the build. A packaged app without a backend cannot
 * start, and an app with the wrong backend -- or with the developer's
 * credential inside it -- is worse than no app at all.
 */
function stageBackendForBuild() {
  const dest = path.join(ROOT, 'build', 'backend');
  let files;
  try {
    files = stageBackend(dest);
  } catch (err) {
    console.error('[build] could not stage the backend:', err.message);
    console.error('[build] The app bundle cannot start a backend it does not contain.');
    process.exit(1);
  }
  console.log(`[build] staged ${files.length} backend file(s) into build/backend`);
}

/**
 * Decide which declared extraResources actually exist.
 *
 * `package.json` copies the staged backend (`build/backend`) and `../.venv`
 * into the app bundle. A missing `from` path is not reported clearly by
 * electron-builder -- it fails
 * partway through packaging with an error that does not name the absent
 * directory, which is why `frontend/release/` had never been produced here with
 * no indication of why.
 *
 * `build/backend` is required: without it the app has no backend to start.
 * It is staged from the repository's source for `backend/` (see
 * `scripts/backend_source_set.js`) before this function runs, so the bundle
 * cannot carry the developer's bearer token or the research database.
 * `build/python` is required too: the packaged app resolves its interpreter
 * only from `resources/python` (see `scripts/python_runtime.js`), and a
 * release that silently borrows the machine's Python is exactly the
 * non-portability this packaging removes.
 *
 * Both entries are therefore required: the build fails rather than producing
 * an installer that cannot start.
 *
 * The resolved list is NOT handed to electron-builder: it reads
 * `extraResources` from package.json itself, and re-supplying them here
 * would copy every file twice (see `effectiveConfig`). An absent *optional*
 * resource needs no handling here -- electron-builder skips a missing
 * extraResource with a warning, which is what "optional" means.
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
    const required = /backend|python/i.test(entry.from);
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
 * electron-builder reads package.json's `build` block and MERGES the config
 * handed to `builder.build()` into it -- `deepAssign` in builder-util
 * concatenates arrays instead of replacing them. Re-supplying `files` and
 * `extraResources` here therefore appended a second copy of every entry, and
 * the packager copied each file twice, concurrently, into the same
 * destination. The second copy of a file the first copy still had open failed
 * with `EBUSY: resource busy or locked`, so the packaging step never
 * produced an installer. (An earlier comment here claimed the opposite --
 * that an overlay replaces rather than merges -- and acting on it is what
 * introduced the duplicate.)
 *
 * So: pass only what package.json does not already declare -- the `win`
 * override and the output directory -- and let electron-builder read `files`
 * and `extraResources` from the `build` block exactly once.
 */
function effectiveConfig() {
  const pkg = require(path.join(ROOT, 'package.json'));
  const base = pkg.build || {};

  const config = {
    directories: Object.assign({}, base.directories, { output: 'release' }),
  };

  const winOverrides = windowsSigningConfig();
  if (Object.keys(winOverrides).length) {
    config.win = Object.assign({}, base.win, winOverrides);
  }
  return config;
}

function stagePythonRuntime() {
  // A release ships its interpreter. `scripts/python_runtime.js` assembles a
  // relocatable CPython + site-packages and executes the result to prove the
  // backend's imports resolve; a failure here stops the build.
  const { stage } = require('./python_runtime');
  let result;
  try {
    result = stage({ force: false });
  } catch (err) {
    console.error('[build] could not stage the bundled Python runtime:', err.message);
    console.error('[build] The installer would have no interpreter to start the backend.');
    process.exit(1);
  }
  const manifest = result.manifest || {};
  console.log(
    `[build] python runtime ${result.skipped ? 'already staged' : 'staged'}: ` +
    `${manifest.version || '?'} -> build/python`
  );
}

function main() {
  stageBackendForBuild();
  stagePythonRuntime();
  const extraResources = resolveExtraResources();
  console.log(
    '[build] extraResources: ' +
    (extraResources.map((e) => `${e.from} -> ${e.to}`).join(', ') || '(none)')
  );
  ensureRendererBuilt();
  const builder = require('electron-builder');
  builder.build({ config: effectiveConfig() })
    .then((paths) => {
      console.log('[build] done:', Array.isArray(paths) ? paths.join(', ') : paths);
    })
    .catch(err => {
      console.error('[build] electron-builder failed:', err);
      process.exit(1);
    });
}

main();
