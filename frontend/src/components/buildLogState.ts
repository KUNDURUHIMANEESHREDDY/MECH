/**
 * Run-identity guard for build-log snapshots (T14).
 *
 * Extracted from `BuildLog.vue` so the guard can be tested without mounting
 * the SFC -- `@vue/test-utils` is not a dependency here, and the defect is the
 * missing check rather than Vue's reactivity.
 *
 * The backend already publishes the identity this keys on:
 * `backend/runtime/build_runner.py` rewrites `started_at` on every
 * `_run_build`, so `started_at` is the run's serial number.
 */

export interface BuildState {
  status?: unknown;
  started_at?: unknown;
  finished_at?: unknown;
  output_tail?: unknown;
  target?: unknown;
}

export interface BuildEntry {
  id: string;
  message: string;
}

/** The view slice this mutates. */
export interface BuildView {
  status: 'idle' | 'running' | 'success' | 'error';
  entries: BuildEntry[];
  runKey: string | null;
  staleIgnored: number;
}

function num(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

/**
 * Identity of a build snapshot: its status and when it started.
 *
 * `started_at` is what distinguishes two builds that share a status. A
 * snapshot with no `started_at` gets an empty stamp and sorts *before* any
 * stamped one, so it can never displace a run it cannot be attributed to.
 */
export function buildSnapshotKey(state: BuildState): string {
  const status = typeof state.status === 'string' ? state.status : 'unknown';
  const started = num(state.started_at);
  return `${status}:${started === null ? '' : started}`;
}

/** Later `started_at` wins; equal stamps are treated as the same build. */
export function newestSnapshotKey(a: string | null, b: string | null): string | null {
  if (a === null) return b;
  if (b === null) return a;
  const stampA = num(a.split(':').pop());
  const stampB = num(b.split(':').pop());
  if (stampA === null) return b;
  if (stampB === null) return a;
  return stampB >= stampA ? b : a;
}

function mapStatus(raw: string, current: string): 'idle' | 'running' | 'success' | 'error' {
  if (raw === 'running') return 'running';
  if (raw === 'completed') return 'success';
  if (raw === 'failed') return 'error';
  if (raw === 'idle' && current !== 'running') return 'idle';
  return current as 'idle' | 'running' | 'success' | 'error';
}

function entriesFromTail(state: BuildState, status: string): BuildEntry[] {
  const tail = typeof state.output_tail === 'string' ? state.output_tail : '';
  if (!tail) return [];
  const stamp =
    (typeof state.finished_at === 'string' && state.finished_at) ||
    (typeof state.started_at === 'string' && state.started_at) ||
    '';
  return tail
    .split(/\r?\n/)
    .filter((line) => line.trim() !== '')
    .slice(-200)
    .map((line, index) => ({
      id: `backend-build-${index}-${line.slice(0, 20)}`,
      message: line,
      // Level is derivable from the applied status by the caller; kept out of
      // the entry so this module stays free of the component's type union.
      level: status === 'error' ? 'error' : 'info',
    })) as BuildEntry[];
}

/** The `started_at` stamp from a snapshot key, or null if it carries none. */
function stampOf(key: string | null): number | null {
  if (key === null) return null;
  return num(key.slice(key.lastIndexOf(':') + 1));
}

/**
 * Apply a backend build snapshot, unless it is older than what the view
 * already shows.
 *
 * Ordering is by `started_at` alone, not by the whole key. Two consequences
 * that a naive key comparison gets wrong:
 *
 *   * A status *advance* within one build (`running` -> `success`) keeps the
 *     same `started_at`, so it must be accepted. Comparing `status` in the
 *     key would read it as a different build.
 *   * A response with no `started_at` carries no evidence about which build it
 *     describes, so it cannot displace a stamped one.
 *
 * Returns `true` when the snapshot was applied. When it returns `false` the
 * caller must not act on the snapshot's status either -- in particular it
 * must not call `stopPolling()` on a stale terminal status, which would
 * abandon a build that is still running.
 */
export function applyBuildSnapshot(view: BuildView, state: BuildState): boolean {
  const key = buildSnapshotKey(state);
  const incoming = stampOf(key);
  const current = stampOf(view.runKey);

  if (view.runKey !== null) {
    if (incoming === null || (current !== null && incoming < current)) {
      view.staleIgnored += 1;
      return false;
    }
  }

  view.status = mapStatus(
    typeof state.status === 'string' ? state.status : '',
    view.status,
  );
  // Keep the newest stamp seen so a later stale response is still rejected.
  view.runKey = newestSnapshotKey(view.runKey, key);

  const entries = entriesFromTail(state, view.status);
  if (entries.length) view.entries = entries;
  return true;
}