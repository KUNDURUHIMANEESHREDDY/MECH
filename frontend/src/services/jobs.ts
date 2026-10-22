/**
 * Client for the backend's persisted job queue (`backend/api/runtime_api.py`,
 * mounted at `/api/v1/jobs` and `/api/v2/jobs`; this client uses v1).
 *
 * Why not `request()` from `services/api.ts`: it throws away the response body,
 * so a 409 ("already finished") and a 503 ("queue full") would both read as
 * "POST /x failed: 409". The queue's refusals carry the backend's own `detail`
 * and callers need it verbatim. The transport is still `apiFetch`, which routes
 * through the Electron preload proxy when present -- a raw `fetch` would bypass
 * the proxy and lose the desktop bearer token.
 *
 * Nothing is defaulted here. A missing field stays missing; the view renders it
 * as "not returned". There is no progress, no ETA and no stage list because the
 * backend sends none -- the only lifecycle facts are the status and timestamps.
 */

import { apiFetch, apiUrl, authHeaders } from './api';

export type JsonRecord = Record<string, unknown>;

/** The only kinds `HANDLERS` in runtime_api.py can run. */
export const JOB_KINDS = ['benchmark_sweep', 'ioi_stability'] as const;
export type JobKind = (typeof JOB_KINDS)[number];

/** `TERMINAL` in runtime_api.py. `queued` and `running` are the non-terminal ones. */
export const TERMINAL_JOB_STATUSES = [
  'completed',
  'failed',
  'cancelled',
  'unavailable',
  'interrupted',
] as const;
export type TerminalJobStatus = (typeof TERMINAL_JOB_STATUSES)[number];
export type JobStatus = 'queued' | 'running' | TerminalJobStatus;

export type JobsErrorCategory =
  | 'unreachable'
  | 'unauthorized'
  | 'invalid_request'
  | 'not_found'
  | 'conflict'
  | 'queue_full'
  | 'http_error';

export interface JobRecord {
  job_id: string;
  kind: string;
  /** runtime_api.py stores submitted params under `payload` (the body calls it `params`). */
  payload: JsonRecord | null;
  status: string;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  result: unknown;
  error: string | null;
  cancel_requested: boolean;
  max_concurrency: number | null;
  persist_error: string | null;
}

export interface JobBounds {
  max_concurrency: number | null;
  max_queued: number | null;
  job_dir: string;
}

export interface JobList {
  /** Newest first, in the order the backend returned. */
  jobs: JobRecord[];
  /** The backend's own `count`; null when it did not send one. */
  count: number | null;
  /** Configured limits reported with the list; null when the body had no `bounds`. */
  bounds: JobBounds | null;
  /** Kinds the backend says it can run; [] when the body had no `kinds`. */
  kinds: string[];
  /** `"live"` when the backend measured this response. */
  provenance: string;
}

export interface SubmittedJob {
  job: JobRecord;
  provenance: string;
}

export interface ListJobsOptions {
  status?: string;
  kind?: string;
  limit?: number;
}

const REQUEST_TIMEOUT_MS = 12_000;
const MAX_RESPONSE_TEXT = 2000;

export class JobsHttpError extends Error {
  readonly method: string;
  readonly path: string;
  /** HTTP status, or 0 when the request never reached the backend. */
  readonly status: number;
  readonly responseText: string;
  /** The backend's `detail` (or `reason`/`error`) verbatim; '' when absent. */
  readonly detail: string;

  constructor(method: string, path: string, status: number, responseText: string) {
    const detail = backendDetail(responseText);
    const summary = status === 0
      ? `${method} ${path} did not complete: ${responseText}`
      : detail
        ? `${method} ${path} failed: HTTP ${status} — ${detail}`
        : `${method} ${path} failed: HTTP ${status}`;
    super(summary);
    this.name = 'JobsHttpError';
    this.method = method;
    this.path = path;
    this.status = status;
    this.responseText = responseText;
    this.detail = detail;
  }

  get category(): JobsErrorCategory {
    if (this.status === 0) return 'unreachable';
    if (this.status === 401 || this.status === 403) return 'unauthorized';
    if (this.status === 400 || this.status === 422) return 'invalid_request';
    if (this.status === 404) return 'not_found';
    if (this.status === 409) return 'conflict';
    if (this.status === 503) return 'queue_full';
    return 'http_error';
  }

  /** True when the queue is full; the job was NOT accepted. */
  get isQueueFull(): boolean {
    return this.status === 503;
  }

  /** True when cancel was refused because the job already finished; NOT a success. */
  get isAlreadyFinished(): boolean {
    return this.status === 409;
  }

  /** True when the API could not be reached at all (no HTTP status). */
  get isUnreachable(): boolean {
    return this.status === 0;
  }
}

export function isJobsHttpError(error: unknown): error is JobsHttpError {
  return error instanceof JobsHttpError;
}

/** Human-readable failure line: status plus the backend's own reason. */
export function describeJobsError(error: unknown): string {
  if (error instanceof JobsHttpError) return error.message;
  if (error instanceof Error && error.message) return error.message;
  return 'The job queue request did not complete.';
}

/** Terminal means the backend will never work on the job again. */
export function isTerminalJobStatus(status: unknown): boolean {
  return typeof status === 'string'
    && (TERMINAL_JOB_STATUSES as readonly string[]).includes(status);
}

export function isActiveJobStatus(status: unknown): boolean {
  return !isTerminalJobStatus(status);
}

export async function listJobs(opts: ListJobsOptions = {}): Promise<JobList> {
  const query = new URLSearchParams();
  if (opts.status) query.set('status', opts.status);
  if (opts.kind) query.set('kind', opts.kind);
  if (typeof opts.limit === 'number') query.set('limit', String(opts.limit));
  const qs = query.toString();
  const body = await requestJson('GET', `/api/v1/jobs${qs ? `?${qs}` : ''}`);

  if (!Array.isArray(body.jobs)) {
    // An absent `jobs` is not an empty queue; refuse instead of rendering zeros.
    throw new JobsHttpError('GET', '/api/v1/jobs', 200,
      'the response did not contain a `jobs` array');
  }
  const jobs = body.jobs.map(normalizeJob).filter((job): job is JobRecord => job !== null);
  return {
    jobs,
    count: asNumber(body.count),
    bounds: asBounds(body.bounds),
    kinds: asStringArray(body.kinds),
    provenance: asString(body.provenance),
  };
}

export async function submitJob(kind: JobKind, params: JsonRecord = {}): Promise<SubmittedJob> {
  if (!(JOB_KINDS as readonly string[]).includes(kind)) {
    throw new TypeError(`unknown job kind ${String(kind)}; one of ${JOB_KINDS.join(', ')}`);
  }
  const body = await requestJson('POST', '/api/v1/jobs', { kind, params });
  const job = normalizeJob(body.job);
  if (!job) {
    throw new JobsHttpError('POST', '/api/v1/jobs', 200,
      'the response did not contain a job record');
  }
  return { job, provenance: asString(body.provenance) };
}

export async function getJob(jobId: string): Promise<SubmittedJob> {
  const path = `/api/v1/jobs/${encodeURIComponent(jobId)}`;
  const body = await requestJson('GET', path);
  const job = normalizeJob(body.job);
  if (!job) {
    throw new JobsHttpError('GET', path, 200, 'the response did not contain a job record');
  }
  return { job, provenance: asString(body.provenance) };
}

export async function cancelJob(jobId: string): Promise<SubmittedJob> {
  const path = `/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`;
  const body = await requestJson('POST', path);
  const job = normalizeJob(body.job);
  if (!job) {
    throw new JobsHttpError('POST', path, 200, 'the response did not contain a job record');
  }
  return { job, provenance: asString(body.provenance) };
}

// --------------------------------------------------------------------------- //
// transport
// --------------------------------------------------------------------------- //

async function requestJson(
  method: string,
  path: string,
  body?: unknown,
): Promise<JsonRecord> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let response: Response;
  try {
    const init: RequestInit = {
      method,
      signal: controller.signal,
      headers: body === undefined
        ? undefined
        : authHeaders({ 'Content-Type': 'application/json' }),
    };
    if (body !== undefined) init.body = JSON.stringify(body);
    response = await apiFetch(apiUrl(path), init);
  } catch (error) {
    const reason = error instanceof DOMException && error.name === 'AbortError'
      ? `timed out after ${REQUEST_TIMEOUT_MS} ms`
      : error instanceof Error ? error.message : String(error);
    throw new JobsHttpError(method, path, 0, reason);
  } finally {
    clearTimeout(timer);
  }

  const text = await response.text();
  if (!response.ok) {
    throw new JobsHttpError(method, path, response.status, text.slice(0, MAX_RESPONSE_TEXT));
  }
  if (!text.trim()) {
    throw new JobsHttpError(method, path, response.status, 'the backend returned an empty body');
  }
  const parsed = tryParseJson(text);
  if (!isRecord(parsed)) {
    throw new JobsHttpError(method, path, response.status, 'the backend response was not a JSON object');
  }
  return parsed;
}

function backendDetail(responseText: string): string {
  const parsed = tryParseJson(responseText);
  if (!isRecord(parsed)) {
    const text = responseText.trim();
    return text && text.length <= 300 ? text : '';
  }
  for (const key of ['detail', 'reason', 'error', 'message']) {
    const value = parsed[key];
    if (typeof value === 'string' && value.trim()) return value.trim();
    if (Array.isArray(value)) {
      const parts = value
        .map((item) => (typeof item === 'string' ? item : JSON.stringify(item)))
        .filter((item) => item && item.trim());
      if (parts.length) return parts.join('; ');
    }
    if (isRecord(value)) {
      const code = asString(value.code);
      const reason = asString(value.reason) || asString(value.message) || asString(value.error);
      if (reason && code) return `${code}: ${reason}`;
      if (reason || code) return reason || code;
    }
  }
  return '';
}

// --------------------------------------------------------------------------- //
// normalization (keeps absent values absent)
// --------------------------------------------------------------------------- //

function normalizeJob(value: unknown): JobRecord | null {
  if (!isRecord(value)) return null;
  const jobId = asString(value.job_id);
  if (!jobId) return null;
  return {
    job_id: jobId,
    kind: asString(value.kind),
    // The persisted record names submitted params `payload`; `params` is the
    // request body's name, so an old record read as `params` would be a guess.
    payload: isRecord(value.payload) ? value.payload : null,
    status: asString(value.status),
    created_at: asString(value.created_at) || null,
    started_at: asString(value.started_at) || null,
    finished_at: asString(value.finished_at) || null,
    result: value.result === undefined ? null : value.result,
    error: asString(value.error) || null,
    cancel_requested: value.cancel_requested === true,
    max_concurrency: asNumber(value.max_concurrency),
    persist_error: asString(value.persist_error) || null,
  };
}

function asBounds(value: unknown): JobBounds | null {
  if (!isRecord(value)) return null;
  return {
    max_concurrency: asNumber(value.max_concurrency),
    max_queued: asNumber(value.max_queued),
    job_dir: asString(value.job_dir),
  };
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string' && item.trim() !== '')
    : [];
}

export function asString(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

export function asNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function tryParseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}
