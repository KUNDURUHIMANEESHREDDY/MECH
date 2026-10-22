import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  JobsHttpError,
  cancelJob,
  getJob,
  listJobs,
  submitJob,
} from '../../src/services/jobs';

/**
 * The transport is the Electron bridge, exactly as `desktopApiClient.test.js`
 * mocks it. `apiFetch` forwards to `window.appApi.apiRequest` when it exists,
 * so these tests exercise the same path the desktop build uses -- and a raw
 * `fetch` would bypass the preload proxy that carries the bearer token.
 */

let requests = [];
let responses = [];

function installBridge(queued) {
  requests = [];
  responses = [...queued];
  window.appApi = {
    apiRequest: vi.fn(async (request) => {
      requests.push(request);
      const next = responses.shift();
      if (!next) throw new Error(`unexpected request: ${JSON.stringify(request)}`);
      return { status: next.status, body: next.body };
    }),
  };
}

function jobRecord(overrides = {}) {
  return {
    job_id: 'j000000000001',
    kind: 'benchmark_sweep',
    payload: { benchmarks: ['IOI'], n_prompts: 5 },
    status: 'queued',
    created_at: '2026-01-01T00:00:00+00:00',
    started_at: null,
    finished_at: null,
    result: null,
    error: null,
    cancel_requested: false,
    max_concurrency: 2,
    ...overrides,
  };
}

async function captureError(promise) {
  try {
    await promise;
    return null;
  } catch (error) {
    return error;
  }
}

afterEach(() => {
  delete window.appApi;
  vi.restoreAllMocks();
});

describe('jobs client', () => {
  it('submitJob posts the documented body shape and keeps the backend job', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch');
    installBridge([{
      status: 202,
      body: { job: jobRecord(), provenance: 'live' },
    }]);

    const result = await submitJob('benchmark_sweep', { benchmarks: ['IOI'], n_prompts: 5 });

    expect(requests).toHaveLength(1);
    expect(requests[0].method).toBe('POST');
    expect(requests[0].path).toBe('/api/v1/jobs');
    expect(requests[0].body).toEqual({
      kind: 'benchmark_sweep',
      params: { benchmarks: ['IOI'], n_prompts: 5 },
    });
    expect(result.provenance).toBe('live');
    expect(result.job.job_id).toBe('j000000000001');
    expect(result.job.status).toBe('queued');
    expect(result.job.payload).toEqual({ benchmarks: ['IOI'], n_prompts: 5 });
    // The bridge carried the request; a raw fetch would skip the preload proxy.
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('turns a cancel 409 into a rejection carrying the backend detail, not a cancellation', async () => {
    installBridge([{
      status: 409,
      body: { detail: 'job is already completed; it cannot be cancelled' },
    }]);

    const error = await captureError(cancelJob('j000000000001'));

    expect(error).toBeInstanceOf(JobsHttpError);
    expect(error.status).toBe(409);
    expect(error.isAlreadyFinished).toBe(true);
    expect(error.detail).toBe('job is already completed; it cannot be cancelled');
    expect(error.message).toMatch(/already completed/);
    expect(requests[0]).toMatchObject({
      method: 'POST',
      path: '/api/v1/jobs/j000000000001/cancel',
    });
    // A rejection means there is no job record the view could mistake for success.
    expect('job' in (error ?? {})).toBe(false);
  });

  it('reports a submit 503 as queue-full, not as an accepted job', async () => {
    installBridge([{
      status: 503,
      body: { detail: 'queue is full: 64 non-terminal job(s), limit 64' },
    }]);

    const error = await captureError(submitJob('ioi_stability', {}));

    expect(error).toBeInstanceOf(JobsHttpError);
    expect(error.status).toBe(503);
    expect(error.isQueueFull).toBe(true);
    expect(error.category).toBe('queue_full');
    expect(error.message).toMatch(/queue is full/);
  });

  it('surfaces a 400 validator message verbatim', async () => {
    installBridge([{
      status: 400,
      body: { detail: 'n_seeds must be an integer between 1 and 10' },
    }]);

    const error = await captureError(submitJob('ioi_stability', { n_seeds: 11 }));

    expect(error.status).toBe(400);
    expect(error.category).toBe('invalid_request');
    expect(error.message).toContain('n_seeds must be an integer between 1 and 10');
  });

  it('listJobs tolerates jobs: [] and invents no entries', async () => {
    const bounds = {
      max_concurrency: 2,
      max_queued: 64,
      job_dir: 'C:/mech/backend/storage/v2_jobs',
    };
    installBridge([{
      status: 200,
      body: {
        jobs: [],
        count: 0,
        bounds,
        kinds: ['benchmark_sweep', 'ioi_stability'],
        provenance: 'live',
      },
    }]);

    const listing = await listJobs({ limit: 100 });

    expect(listing.jobs).toEqual([]);
    expect(listing.count).toBe(0);
    expect(listing.bounds).toEqual(bounds);
    expect(listing.kinds).toEqual(['benchmark_sweep', 'ioi_stability']);
    expect(listing.provenance).toBe('live');

    const listingCall = requests[0];
    expect(listingCall.method).toBe('GET');
    expect(listingCall.path).toBe('/api/v1/jobs');
    expect(listingCall.query).toEqual({ limit: '100' });
  });

  it('refuses a listing without a jobs array instead of rendering an empty queue', async () => {
    installBridge([{ status: 200, body: { count: 0, provenance: 'live' } }]);

    const error = await captureError(listJobs());

    expect(error).toBeInstanceOf(JobsHttpError);
    expect(error.message).toMatch(/jobs/);
  });

  it('refuses a kind the backend does not implement without sending it', async () => {
    installBridge([]);

    const error = await captureError(submitJob('definitely_not_a_kind', {}));

    expect(error).toBeInstanceOf(TypeError);
    expect(requests).toEqual([]);
  });

  it('getJob requests the job by id and keeps the backend provenance', async () => {
    installBridge([{
      status: 200,
      body: { job: jobRecord({ status: 'cancelled', error: 'cancelled before it started' }), provenance: 'live' },
    }]);

    const result = await getJob('j000000000001');

    expect(requests[0]).toMatchObject({ method: 'GET', path: '/api/v1/jobs/j000000000001' });
    expect(result.job.status).toBe('cancelled');
    expect(result.job.error).toBe('cancelled before it started');
    expect(result.provenance).toBe('live');
  });
});
