'use strict';

/**
 * The backend source set: the files a packaged backend is allowed to contain.
 *
 * The Electron build copies a copy of `backend/` into the app bundle, and
 * the standalone build copies one too. What went into that copy used to be
 * "whatever is on disk", which put the developer's control-plane bearer
 * token and the research database into every installer, and shipped a
 * backend that silently differed from the source tree.
 *
 * The rule now: a packaged backend contains exactly the repository's source
 * for `backend/` -- the files git tracks, plus the untracked files git does
 * not ignore -- minus the secrets and runtime state that must never ship.
 *
 * Git is the only thing that can tell those apart. A `.pyc` and a
 * `.dsh-ops-update-*.lock` file left behind by a tool run are untracked
 * noise. `mech.db`, `.mech_api_token`, the evidence directory and the SAE
 * checkpoints are deliberately untracked state. A filesystem walk cannot
 * separate the three, and a copy of one of them is a shipped credential.
 *
 * This is the same rule as `scripts/packaged_backend_guard.py`, which
 * compares a packaged tree against it. The two implementations are kept in
 * step by `tests/pytest/test_packaged_backend_drift.py`, which asserts that
 * they produce the same file list.
 */

const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.join(__dirname, '..', '..');
const FRONTEND = path.join(__dirname, '..');

/** The backend directory, relative to the repository root. */
const BACKEND = 'backend';

/** Where the build stages the backend, relative to `frontend/`. */
const STAGED = path.join('build', 'backend');

/**
 * Secrets and runtime state that must never reach a packaged tree.
 *
 * Must match FORBIDDEN_PATTERNS in scripts/packaged_backend_guard.py --
 * the test asserts that they agree, so a change to one without the other
 * fails the suite rather than shipping a credential.
 */
const FORBIDDEN = [
  'storage/.mech_api_token',
  'storage/*.db*',
  'storage/evidence/**',
  'storage/v2_jobs/**',
  '*.db*',
  '*.lock',
];

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Translate one electron-builder filter glob into a regular expression.
 *
 * Supports the subset the build filter uses: `**` (any number of path
 * segments), `*` (any run of characters inside one segment) and `?`.
 */
function globToRegExp(pattern) {
  let out = '';
  let i = 0;
  while (i < pattern.length) {
    if (pattern.startsWith('**/', i)) {
      out += '(?:[^/]+/)*';
      i += 3;
    } else if (pattern.startsWith('/**', i) && i + 3 === pattern.length) {
      out += '/.*';
      i += 3;
    } else if (pattern.startsWith('**', i) && i + 2 === pattern.length) {
      out += '.*';
      i += 2;
    } else if (pattern[i] === '*') {
      out += '[^/]*';
      i += 1;
    } else if (pattern[i] === '?') {
      out += '[^/]';
      i += 1;
    } else {
      out += escapeRegExp(pattern[i]);
      i += 1;
    }
  }
  return new RegExp(`^${out}$`);
}

/**
 * Translate one shell-style name pattern into a regular expression.
 *
 * Same shape as Python's fnmatch: `*` matches any run of characters, `?`
 * matches one, and `[...]` is a character class.
 */
function fnmatchToRegExp(pattern) {
  let out = '';
  let i = 0;
  while (i < pattern.length) {
    const c = pattern[i];
    if (c === '*') {
      out += '.*';
      i += 1;
    } else if (c === '?') {
      out += '.';
      i += 1;
    } else if (c === '[') {
      let j = i + 1;
      if (pattern[j] === '!') j += 1;
      if (pattern[j] === ']') j += 1;
      while (j < pattern.length && pattern[j] !== ']') j += 1;
      if (j >= pattern.length) {
        out += '\\[';
        i += 1;
      } else {
        let cls = pattern.slice(i + 1, j).replace(/\\/g, '\\\\');
        if (cls[0] === '!') cls = `^${cls.slice(1)}`;
        out += `[${cls}]`;
        i = j + 1;
      }
    } else {
      out += escapeRegExp(c);
      i += 1;
    }
  }
  return new RegExp(`^${out}$`);
}

/**
 * Whether `relPath` (POSIX, relative to the backend root) matches `pattern`.
 *
 * A pattern without a `/` matches at any depth, which is how
 * electron-builder reads a bare name: `!__pycache__` excludes every
 * `__pycache__` directory in the tree, not only one at the root. The
 * comparison is case-insensitive on Windows, which is the direction a
 * "never ship a secret" rule wants to err in.
 */
function patternMatches(pattern, relPath) {
  if (!pattern.includes('/')) {
    const fold = process.platform === 'win32';
    const regex = fnmatchToRegExp(fold ? pattern.toLowerCase() : pattern);
    return relPath.split('/').some((part) => (
      regex.test(fold ? part.toLowerCase() : part)
    ));
  }
  return globToRegExp(pattern).test(relPath);
}

/**
 * The filter electron-builder applies to the backend copy.
 *
 * Found by its destination (`"to": "backend"`), not by its source: the
 * source is now the staged build directory, and the filter is the part
 * that must not change.
 */
function backendFilter() {
  const pkg = JSON.parse(fs.readFileSync(path.join(FRONTEND, 'package.json'), 'utf8'));
  const entries = (pkg.build && pkg.build.extraResources) || [];
  for (const entry of entries) {
    if (entry && entry.to === 'backend') return entry.filter || [];
  }
  throw new Error(
    'frontend/package.json declares no extraResources entry with "to": "backend"',
  );
}

/**
 * Whether `relPath` must be kept out of a packaged backend.
 *
 * The build filter is the first rule; the forbidden list is the second, and
 * it does not depend on the filter being right: a typo in the filter must
 * not become a shipped credential.
 */
function isExcluded(relPath, filter) {
  let excluded = false;
  for (const pattern of filter) {
    if (pattern.startsWith('!')) {
      if (patternMatches(pattern.slice(1), relPath)) excluded = true;
    } else if (patternMatches(pattern, relPath)) {
      excluded = false;
    }
  }
  for (const pattern of FORBIDDEN) {
    if (patternMatches(pattern, relPath)) excluded = true;
  }
  return excluded;
}

/**
 * The files git knows about under `backend/`.
 *
 * `--cached --others --exclude-standard` lists the tracked files and the
 * untracked ones git does not ignore. `--exclude-standard` is what keeps
 * the runtime state -- the database, the bearer token, the evidence
 * directory -- out, while the tracked files stay in.
 */
function gitSourceFiles() {
  const res = spawnSync(
    'git',
    ['ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', BACKEND],
    { cwd: ROOT, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 },
  );
  if (res.error) {
    throw new Error(`could not run git: ${res.error.message}`);
  }
  if (res.status !== 0) {
    throw new Error(`git ls-files failed with code ${res.status}: ${res.stderr || ''}`);
  }
  return res.stdout.split('\0').filter(Boolean);
}

/**
 * The source set: the relative paths under `backend/` that a packaged
 * backend may contain.
 */
function sourceSet() {
  const filter = backendFilter();
  const out = [];
  for (const name of gitSourceFiles()) {
    const rel = name.replace(/\\/g, '/');
    if (!rel.startsWith(`${BACKEND}/`)) continue;
    const relPath = rel.slice(BACKEND.length + 1);
    if (isExcluded(relPath, filter)) continue;
    let isFile = false;
    try {
      isFile = fs.statSync(path.join(ROOT, BACKEND, relPath)).isFile();
    } catch (err) {
      isFile = false;
    }
    // Tracked but deleted from the working tree: nothing to copy.
    if (!isFile) continue;
    out.push(relPath);
  }
  return out.sort();
}

/**
 * Stage the backend source set into `destDir`.
 *
 * The destination is replaced, not merged: a file left over from an earlier
 * build is a file the source does not have, and the packaged backend must
 * not contain it.
 */
function stageBackend(destDir) {
  const files = sourceSet();
  fs.rmSync(destDir, { recursive: true, force: true });
  fs.mkdirSync(destDir, { recursive: true });
  for (const rel of files) {
    const from = path.join(ROOT, BACKEND, rel);
    const to = path.join(destDir, rel);
    fs.mkdirSync(path.dirname(to), { recursive: true });
    fs.copyFileSync(from, to);
  }
  return files;
}

module.exports = {
  BACKEND,
  FORBIDDEN,
  ROOT,
  STAGED,
  backendFilter,
  isExcluded,
  patternMatches,
  sourceSet,
  stageBackend,
};
