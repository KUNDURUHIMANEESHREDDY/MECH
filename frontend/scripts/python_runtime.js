'use strict';

/**
 * Stage a portable CPython runtime for the Windows installer.
 *
 * Why not ship `../.venv`?
 * ------------------------
 * A virtual environment is a set of *pointers*: `pyvenv.cfg` records the prefix
 * it was created against and the `Scripts/*.exe` shims bake absolute paths.
 * Copying one to another machine yields an interpreter that either cannot find
 * its stdlib or silently resolves the system Python. That is why the previous
 * `extraResources` entry was dropped at build time, leaving the packaged app to
 * start whatever `python` happened to be on PATH.
 *
 * What this produces
 * ------------------
 * `frontend/build/python/` -- a directory that *is* the distribution:
 *
 *   python.exe, python3XX.dll, DLLs/, Lib/   the build machine's CPython 3.11
 *   Lib/site-packages/                       that installation's dependencies
 *   MECH-RUNTIME.json                        what was staged, from what
 *
 * A copied full CPython installation resolves `Lib` and `DLLs` relative to
 * `python.exe`, so the directory is relocatable: no `pyvenv.cfg`, no absolute
 * path inside it.
 *
 * One environment, not a merge
 * ----------------------------
 * Everything is copied from a single interpreter's *base prefix*. Merging two
 * site-packages directories looks convenient and produces a runtime with two
 * torch versions and one set of native DLLs -- an import that either fails or,
 * worse, works until it does not. If the chosen interpreter cannot import the
 * backend stack, staging fails and says so.
 *
 * Usage
 * -----
 *   node scripts/python_runtime.js              # assemble from this machine
 *   node scripts/python_runtime.js --reinstall  # pip install --target (network)
 *   node scripts/python_runtime.js --force      # ignore the stamp, restage
 *   node scripts/python_runtime.js --cpu        # CPU-only torch (implies --reinstall)
 *   node scripts/python_runtime.js --plan       # print what would happen
 */

const { spawnSync } = require('node:child_process');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.join(__dirname, '..');
const REPO_ROOT = path.join(ROOT, '..');
const DEST = path.join(ROOT, 'build', 'python');
const MANIFEST = path.join(DEST, 'MECH-RUNTIME.json');
const REQUIREMENTS = path.join(REPO_ROOT, 'requirements.txt');

/**
 * CPU-only torch index.
 *
 * The default wheel for this stack is the CUDA build (several GiB of `nvidia*`
 * runtime libraries). The backend runs inference on whatever device is present
 * and falls back to CPU, so a release that only needs inference can ship the
 * CPU wheel and lose ~5 GiB of installer.
 */
const CPU_TORCH_INDEX = 'https://download.pytorch.org/whl/cpu';

/** Imports the backend cannot start without. */
const REQUIRED_IMPORTS = [
  'fastapi',
  'uvicorn',
  'torch',
  'transformer_lens',
  'numpy',
  'pandas',
];

/**
 * Directories pruned at every depth.
 *
 * Deliberately short: robocopy's /XD is case-insensitive, so a broad name like
 * `tools` also deletes `pandas/core/tools` and breaks the import it was meant
 * to slim. Only names that cannot be a live package subdirectory belong here.
 */
const SKIP_DIRS = ['__pycache__', 'test', 'tests'];

/**
 * Directories pruned only at the top of the distribution: CPython's own
 * extras. They are removed after the copy so the exclusion stays depth-scoped.
 */
const ROOT_PRUNE_DIRS = [
  'Tools',
  'Doc',
  'Scripts', // console-script shims carry absolute paths
  'Lib2to3',
  'idlelib',
  'tkinter',
  'turtledemo',
  'include', // C headers: build-time only
  'share',
];

/** Files that are never needed at import time. */
const SKIP_FILES = ['*.pyc', '*.pyo', '*.pdb', '*.lib', '*.exp', '*.a'];

function pythonExe(prefix) {
  return process.platform === 'win32'
    ? path.join(prefix, 'python.exe')
    : path.join(prefix, 'bin', 'python3');
}

function probe(python, code, timeout = 600000) {
  const res = spawnSync(python, ['-c', code], { encoding: 'utf8', timeout });
  if (res.error) throw new Error(`cannot run ${python}: ${res.error.message}`);
  if (res.status !== 0) {
    throw new Error(`${python} -c ${JSON.stringify(code)} failed:\n${(res.stderr || '').trim()}`);
  }
  return res.stdout.trim();
}

function basePrefix(python) {
  return probe(python, 'import sys; print(sys.base_prefix)', 60000);
}

function runtimeVersion(python) {
  return probe(
    python,
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")',
    60000
  );
}

function requirementsHash() {
  try {
    return crypto.createHash('sha256').update(fs.readFileSync(REQUIREMENTS)).digest('hex');
  } catch {
    return null;
  }
}

/**
 * Copy a tree.
 *
 * robocopy on Windows: an order of magnitude faster than a recursive copy in
 * Node for a multi-gigabyte tree, and this runs on every release build.
 */
function copyTree(from, to) {
  if (!fs.existsSync(from)) throw new Error(`missing source directory: ${from}`);
  fs.mkdirSync(to, { recursive: true });

  if (process.platform === 'win32') {
    const args = [
      from, to,
      '/E', '/MT:16', '/NFL', '/NDL', '/NJH', '/NJS', '/NP',
      '/XD', ...SKIP_DIRS,
      '/XF', ...SKIP_FILES,
    ];
    const res = spawnSync('robocopy', args, { encoding: 'utf8' });
    // robocopy: 0-7 are success codes, 8+ are failures.
    if (res.error) throw new Error(`robocopy failed: ${res.error.message}`);
    if (res.status === null || res.status >= 8) {
      throw new Error(`robocopy ${from} -> ${to} failed with code ${res.status}`);
    }
    return;
  }

  fs.cpSync(from, to, {
    recursive: true,
    dereference: true,
    filter: (src) => {
      if (src === from) return true;
      const name = path.basename(src);
      if (SKIP_DIRS.includes(name)) return false;
      if (SKIP_FILES.some((pattern) => name.endsWith(pattern.replace('*', '')))) return false;
      return true;
    },
  });
}

function installedSizeBytes(dir) {
  let total = 0;
  const walk = (current) => {
    let entries;
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.isFile()) {
        try {
          total += fs.statSync(full).size;
        } catch {
          /* raced with the filesystem; the size is approximate anyway */
        }
      }
    }
  };
  walk(dir);
  return total;
}

/**
 * Remove CUDA runtime libraries left behind by a CUDA torch install.
 *
 * Only meaningful for the CPU profile: there the CUDA libraries are dead weight,
 * and shipping them is what made the first staged runtime 5.6 GiB.
 */
function pruneCudaRuntime(dest) {
  let removed = 0;
  for (const name of fs.readdirSync(dest)) {
    if (!/^(nvidia|cudnn|nccl|triton)/i.test(name)) continue;
    fs.rmSync(path.join(dest, name), { recursive: true, force: true });
    removed += 1;
  }
  return removed;
}

function installCpuTorch(python, siteTarget) {
  const res = spawnSync(
    python,
    [
      '-m', 'pip', 'install',
      '--target', siteTarget,
      '--upgrade', '--force-reinstall', '--no-deps',
      '--index-url', CPU_TORCH_INDEX,
      'torch',
    ],
    { encoding: 'utf8', stdio: 'inherit' }
  );
  if (res.status !== 0) {
    throw new Error(`installing the CPU torch wheel failed with code ${res.status}`);
  }
}

/** Embeddable distributions carry a `python3XX._pth` that overrides sys.path. */
function ensureSitePackagesOnPath(dest) {
  const pth = fs.readdirSync(dest).find((name) => /^python\d+\._pth$/i.test(name));
  if (!pth) return null;
  const target = path.join(dest, pth);
  const lines = fs.readFileSync(target, 'utf8').split(/\r?\n/).filter(Boolean);
  if (!lines.some((line) => line.replace(/[\\/]+$/, '').endsWith('site-packages'))) {
    lines.push('Lib\\site-packages');
  }
  if (!lines.includes('import site')) lines.push('import site');
  fs.writeFileSync(target, `${lines.join('\n')}\n`, 'utf8');
  return pth;
}

/**
 * Execute the staged interpreter and import what the backend imports.
 *
 * One module per subprocess: a single combined import cannot tell you *which*
 * dependency broke, and the point of this step is a diagnosable failure instead
 * of an installer that dies on a user's machine.
 */
function verify(dest) {
  const exe = pythonExe(dest);
  if (!fs.existsSync(exe)) {
    return { ok: false, results: [{ module: '(interpreter)', ok: false, error: `missing: ${exe}` }] };
  }

  const results = [];
  for (const module of REQUIRED_IMPORTS) {
    const res = spawnSync(exe, ['-c', `import ${module}`], { encoding: 'utf8', timeout: 900000 });
    if (res.error) {
      results.push({ module, ok: false, error: res.error.message });
    } else if (res.status !== 0) {
      results.push({
        module,
        ok: false,
        error: (res.stderr || '').trim().split('\n').slice(-3).join('\n'),
      });
    } else {
      results.push({ module, ok: true });
    }
  }
  return { ok: results.every((r) => r.ok), results };
}

function readManifest() {
  try {
    return JSON.parse(fs.readFileSync(MANIFEST, 'utf8'));
  } catch {
    return null;
  }
}

function isFresh(manifest, python, mode, cpu) {
  if (!manifest || !fs.existsSync(pythonExe(DEST))) return false;
  return (
    manifest.python === python &&
    manifest.mode === mode &&
    Boolean(manifest.cpu) === Boolean(cpu) &&
    manifest.requirementsSha256 === requirementsHash() &&
    manifest.version === runtimeVersion(python)
  );
}

/**
 * Which interpreter to stage from.
 *
 * Candidates are probed, not assumed: this checkout's `.venv` holds torch but
 * not fastapi, so "the venv exists" is not evidence that it can run the backend.
 * Only `fastapi` is imported during the probe (it is fast); the full stack is
 * checked by `verify()` after staging.
 */
function resolveBuildPython() {
  const candidates = [];
  if (process.env.MECH_PYTHON) candidates.push(process.env.MECH_PYTHON);
  candidates.push(path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe'));
  candidates.push(path.join(REPO_ROOT, 'venv', 'Scripts', 'python.exe'));
  candidates.push('python');

  const attempts = [];
  for (const candidate of candidates) {
    if (candidate !== 'python' && !fs.existsSync(candidate)) continue;
    const res = spawnSync(candidate, ['-c', 'import fastapi'], { encoding: 'utf8', timeout: 300000 });
    if (res.status === 0) return candidate;
    attempts.push({
      candidate,
      error: (res.error && res.error.message) ||
        (res.stderr || '').trim().split('\n').pop() ||
        `exit ${res.status}`,
    });
  }

  throw new Error(
    'no interpreter on this machine can import the backend requirements.\n' +
    attempts.map((a) => `  ${a.candidate}: ${a.error}`).join('\n') +
    '\nInstall requirements.txt into one of them, or set MECH_PYTHON.'
  );
}

/**
 * Stage the runtime.
 *
 * `installed` copies the base prefix as it is: offline, and coherent because
 * everything comes from one installation. `reinstall` resolves
 * `requirements.txt` into the staged site-packages with pip, for a release that
 * must not inherit the build machine's resolution.
 */
/**
 * The pip invocation that resolves `requirements.txt` into the staged tree.
 *
 * The CPU index is an EXTRA index, not the only one. pip compares the
 * candidates and a local version sorts above the plain release, so
 * `2.14.1+cpu` wins over `2.14.1` and the CPU wheel is the one that gets
 * installed -- while every other dependency still resolves against PyPI.
 *
 * Without it, the CPU profile downloaded the CUDA build (and the gigabytes of
 * `nvidia*` libraries that come with it) and then threw it away, which is how
 * a release build spent an hour downloading a wheel it did not ship.
 *
 * Exported so the profile's index handling can be asserted without
 * downloading anything.
 */
function pipInstallArgs(siteTarget, { cpu = false } = {}) {
  const args = [
    '-m', 'pip', 'install', '--target', siteTarget, '--requirement', REQUIREMENTS,
  ];
  if (cpu) {
    args.splice(args.indexOf('--requirement'), 0, '--extra-index-url', CPU_TORCH_INDEX);
  }
  return args;
}

function stage({ python = resolveBuildPython(), mode = 'installed', force = false, cpu = false } = {}) {
  const existing = readManifest();
  if (!force && isFresh(existing, python, mode, cpu)) {
    return { skipped: true, dest: DEST, manifest: existing };
  }

  fs.rmSync(DEST, { recursive: true, force: true });
  fs.mkdirSync(DEST, { recursive: true });

  const base = basePrefix(python);
  const version = runtimeVersion(python);
  copyTree(base, DEST);

  for (const name of ROOT_PRUNE_DIRS) {
    fs.rmSync(path.join(DEST, name), { recursive: true, force: true });
  }

  const siteTarget = path.join(DEST, 'Lib', 'site-packages');
  if (cpu && mode !== 'reinstall') {
    // An installed CUDA build cannot be converted offline: the CPU wheel has to
    // be downloaded, and silently shipping the CUDA one while claiming `--cpu`
    // would be worse than failing.
    throw new Error('--cpu requires --reinstall: the CPU wheel must be downloaded');
  }
  if (mode === 'reinstall') {
    fs.rmSync(siteTarget, { recursive: true, force: true });
    const res = spawnSync(
      python,
      pipInstallArgs(siteTarget, { cpu }),
      { encoding: 'utf8', stdio: 'inherit' },
    );
    if (res.status !== 0) {
      throw new Error(`pip install --target failed with code ${res.status}`);
    }
    if (cpu) {
      installCpuTorch(python, siteTarget);
      const removed = pruneCudaRuntime(siteTarget);
      if (removed) console.log(`[python_runtime] pruned ${removed} CUDA runtime package(s)`);
    }
  }

  const pth = ensureSitePackagesOnPath(DEST);
  const check = verify(DEST);
  if (!check.ok) {
    fs.rmSync(DEST, { recursive: true, force: true });
    const failed = check.results.filter((r) => !r.ok);
    throw new Error(
      'the staged interpreter cannot import the backend requirements:\n' +
      failed.map((r) => `  ${r.module}: ${r.error}`).join('\n')
    );
  }

  const sizeBytes = installedSizeBytes(DEST);
  const manifest = {
    version,
    python,
    mode,
    cpu,
    // The profile in words, so the record does not depend on the reader
    // knowing that `mode: reinstall` + `cpu: true` means "CPU-only torch".
    profile: cpu ? 'cpu' : 'default',
    sizeBytes,
    basePrefix: base,
    sitePackages: siteTarget,
    requirementsSha256: requirementsHash(),
    pthFile: pth,
    imports: REQUIRED_IMPORTS,
    createdAt: new Date().toISOString(),
  };
  fs.writeFileSync(MANIFEST, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
  console.log(
    `[python_runtime] staged size: ${(sizeBytes / 1024 ** 3).toFixed(2)} GiB`
  );

  return { skipped: false, dest: DEST, manifest };
}

/**
 * The torch profile a build must ship.
 *
 * `MECH_TORCH_PROFILE` makes the choice explicit:
 *
 *   `cpu`   stage (or keep) the CPU-only profile: pip-resolved requirements
 *           plus the CPU torch wheel from download.pytorch.org/whl/cpu.
 *   `auto`  (default) keep whatever profile is already staged, and stage the
 *           default 'installed' profile only when nothing is staged at all.
 *
 * Why it has to be explicit: the default wheel on PyPI is the CUDA build,
 * which drags in several GiB of `nvidia*` runtime libraries. A bundle built
 * that way is ~5.8 GB, which NSIS cannot pack (makensis.exe is 32-bit and
 * cannot memory-map an archive over 2 GB). The profile is recorded in
 * MECH-RUNTIME.json, so the artifact says which torch it carries.
 *
 * The rule that matters: a profile is never changed silently. A staged CPU
 * runtime stays CPU, and a staged default runtime stays default, unless the
 * environment names a different one.
 */
function resolveProfile(existing, env = process.env) {
  const raw = String(env.MECH_TORCH_PROFILE || 'auto').trim().toLowerCase();
  if (raw === 'cpu') return { mode: 'reinstall', cpu: true, explicit: true };
  if (raw !== 'auto') {
    throw new Error(`unknown MECH_TORCH_PROFILE "${raw}": expected "cpu" or "auto"`);
  }
  // No explicit request: keep the profile that is already staged. Silently
  // replacing a staged CPU runtime with the build machine's CUDA one is the
  // exact failure this guards against.
  if (existing && existing.mode) {
    return { mode: existing.mode, cpu: Boolean(existing.cpu), explicit: false };
  }
  return { mode: 'installed', cpu: false, explicit: false };
}

function plan(options = {}) {
  // A staged runtime records the interpreter it came from, so planning is
  // instant after the first build.
  const manifest = readManifest();
  const python = options.python || (manifest && manifest.python) || resolveBuildPython();
  const base = basePrefix(python);
  return {
    dest: DEST,
    manifest: MANIFEST,
    python,
    version: runtimeVersion(python),
    basePrefix: base,
    sitePackages: path.join(base, 'Lib', 'site-packages'),
    requirements: REQUIREMENTS,
    requiredImports: REQUIRED_IMPORTS,
    mode: options.mode || 'installed',
    cpu: Boolean(options.cpu),
    cpuTorchIndex: options.cpu ? CPU_TORCH_INDEX : null,
  };
}

function main(argv) {
  const args = new Set(argv.slice(2));
  const cpu = args.has('--cpu');
  const mode = args.has('--reinstall') || cpu ? 'reinstall' : 'installed';

  if (args.has('--plan')) {
    console.log(JSON.stringify(plan({ mode, cpu }), null, 2));
    return;
  }
  const result = stage({ mode, cpu, force: args.has('--force') });
  if (result.skipped) {
    console.log(`[python_runtime] up to date: ${result.dest} (${result.manifest.version})`);
  } else {
    console.log(
      `[python_runtime] staged ${result.manifest.version} -> ${result.dest} ` +
      `(imports verified: ${result.manifest.imports.join(', ')})`
    );
  }
}

if (require.main === module) {
  try {
    main(process.argv);
  } catch (err) {
    console.error(`[python_runtime] ${err.message}`);
    process.exit(1);
  }
}

module.exports = {
  CPU_TORCH_INDEX,
  DEST,
  MANIFEST,
  REQUIRED_IMPORTS,
  isFresh,
  pipInstallArgs,
  plan,
  readManifest,
  resolveProfile,
  stage,
  verify,
};
