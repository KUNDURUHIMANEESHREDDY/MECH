/**
 * T14: a stale response must not overwrite the current experiment state.
 *
 * `BuildLog.vue` polls `api.getBackendBuild()` every 5s (`startPolling`,
 * line 373) and applies whatever comes back via `applyApiBuildState`, which
 * writes `buildStatus.value` and rebuilds `buildEntries.value` wholesale.
 *
 * The defect, in the component's own shape:
 *
 *   async function refreshApiBuild() {
 *     try {
 *       const result = await api.getBackendBuild();
 *       if (isRecord(result) && isRecord(result.build)) {
 *         apiActive.value = true;
 *         applyApiBuildState(result.build);      // <-- unconditional
 *       }
 *     } catch {
 *       // poll failures must not wipe a running build view; next tick retries
 *     }
 *   }
 *
 * Nothing checks that the response describes the build now in view. The
 * backend already publishes exactly the identity needed --
 * `backend/runtime/build_runner.py` rewrites `started_at` on every
 * `_run_build` -- and the component already reads `started_at` for the
 * per-entry timestamp (line 331) while never comparing it across responses.
 *
 * The interleaving, concretely:
 *
 *   t=0.0  poll fires, request R1 in flight, snapshot of build A
 *   t=0.1  R1 is slow
 *   t=2.0  user starts build B; started_at becomes B.started_at
 *   t=2.1  R1 resolves -- carrying build A's state
 *   t=2.1  applyApiBuildState(A): buildStatus <- A's status,
 *          buildEntries <- A's output_tail
 *
 * The view shows build A's output attributed to build B. And when A finished
 * while B is still running, A's terminal status lands and `stopPolling()`
 * is called (line 339) -- the UI stops watching a build that is running.
 *
 * For a research IDE this is the frontend face of the rule the backend
 * enforces elsewhere: a result must be attributable to the run that produced
 * it. A stale response shown as current is a mis-attributed result.
 *
 * These drive the extracted `applyBuildSnapshot` helper rather than mounting
 * the SFC: the defect is the missing identity check, and mounting would test
 * Vue's reactivity instead. `@vue/test-utils` is not a dependency here, and
 * this suite imports plain modules only -- as the other four tests do.
 */
import { describe, expect, it } from 'vitest';
import {
  applyBuildSnapshot,
  buildSnapshotKey,
  newestSnapshotKey,
} from '../../src/components/buildLogState';

describe('build log snapshot identity', () => {
  it('derives a stable identity from the backend build state', () => {
    const state = { status: 'running', started_at: 1000, target: 'renderer' };
    expect(buildSnapshotKey(state)).toBe('running:1000');
  });

  it('distinguishes two builds that share a status but not a start time', () => {
    const a = buildSnapshotKey({ status: 'running', started_at: 1000 });
    const b = buildSnapshotKey({ status: 'running', started_at: 2000 });
    expect(a).not.toBe(b);
  });
});

describe('applyBuildSnapshot: stale responses are refused', () => {
  /** The minimal view state applyBuildSnapshot mutates. */
  function makeView() {
    return {
      status: 'running',
      entries: [{ id: 'a', message: 'A output' }],
      runKey: null,
      staleIgnored: 0,
    };
  }

  it('accepts the first snapshot it is given', () => {
    const view = makeView();
    const applied = applyBuildSnapshot(view, {
      status: 'running',
      started_at: 1000,
      output_tail: 'A output',
    });

    expect(applied).toBe(true);
    expect(view.status).toBe('running');
    expect(view.runKey).toBe('running:1000');
    expect(view.staleIgnored).toBe(0);
  });

  it('refuses a snapshot older than the one already applied', () => {
    const view = makeView();

    // Newer build B arrives first.
    applyBuildSnapshot(view, {
      status: 'running',
      started_at: 2000,
      output_tail: 'B output',
    });
    expect(view.status).toBe('running');
    expect(view.entries.map((e) => e.message)).toEqual(['B output']);

    // The slow request for build A now lands.
    const applied = applyBuildSnapshot(view, {
      status: 'success',
      started_at: 1000,
      output_tail: 'A output',
    });

    expect(applied).toBe(false);
    expect(view.staleIgnored).toBe(1);
    // The view must still show B -- status, output, and all.
    expect(view.status).toBe('running');
    expect(view.entries.map((e) => e.message)).toEqual(['B output']);
  });

  it('refuses a stale snapshot that would otherwise stop the poller', () => {
    const view = makeView();

    applyBuildSnapshot(view, { status: 'running', started_at: 2000 });
    expect(view.status).toBe('running');

    // Build A's terminal state arrives late. Applied, it flips status to
    // 'success', which makes the component call stopPolling() and abandon
    // build B, which is still running.
    applyBuildSnapshot(view, { status: 'success', started_at: 1000 });

    expect(view.status).toBe('running');
  });

  it('still accepts a newer snapshot for the same build', () => {
    const view = makeView();

    applyBuildSnapshot(view, {
      status: 'running',
      started_at: 2000,
      output_tail: 'line 1',
    });
    applyBuildSnapshot(view, {
      status: 'running',
      started_at: 2000,
      output_tail: 'line 1\nline 2',
    });

    expect(view.staleIgnored).toBe(0);
    expect(view.entries.length).toBe(2);
  });

  it('accepts a terminal snapshot for the same build it is watching', () => {
    const view = makeView();

    applyBuildSnapshot(view, { status: 'running', started_at: 2000 });
    // Backend vocabulary is 'completed'/'failed', not 'success'/'error' --
    // `build_runner.py` line 80. An earlier draft of this test asserted
    // against 'success', which the mapper correctly ignored, so the view
    // stayed 'running' and the test failed for the right reason against the
    // wrong wire format.
    const applied = applyBuildSnapshot(view, {
      status: 'completed',
      started_at: 2000,
    });

    expect(applied).toBe(true);
    expect(view.status).toBe('success');
    expect(view.staleIgnored).toBe(0);
  });

  it('does not let an unstamped snapshot overwrite a stamped one', () => {
    const view = makeView();

    applyBuildSnapshot(view, {
      status: 'running',
      started_at: 2000,
      output_tail: 'B output',
    });

    // A response with no started_at cannot be shown to belong to build B.
    // Refusing is the conservative choice: an unidentifiable snapshot is not
    // evidence about the current run.
    const applied = applyBuildSnapshot(view, {
      status: 'error',
      output_tail: 'mystery',
    });

    expect(applied).toBe(false);
    expect(view.staleIgnored).toBe(1);
    expect(view.status).toBe('running');
    expect(view.entries.map((e) => e.message)).toEqual(['B output']);
  });

  it('reports the newest key it has seen', () => {
    expect(newestSnapshotKey(null, 'running:2000')).toBe('running:2000');
    expect(newestSnapshotKey('running:2000', 'running:1000')).toBe('running:2000');
    expect(newestSnapshotKey('running:1000', 'running:2000')).toBe('running:2000');
  });
});