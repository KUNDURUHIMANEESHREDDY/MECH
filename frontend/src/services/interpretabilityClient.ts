/**
 * Thin client for the interpretability endpoints the four new surfaces use.
 *
 * Differences from `services/api.ts`, and why they matter here:
 *
 *  - The HTTP status is preserved on failure. A 404/405/501 means the backend
 *    does not implement the endpoint; that is an explicit "not available"
 *    state, not a generic error next to an empty chart.
 *  - The response body is kept so a structured backend reason (`detail`,
 *    `error`, `reason`) can be shown verbatim instead of being flattened into
 *    "GET /x failed: 400".
 *  - Nothing is defaulted. A missing number stays `null` and formats as "—";
 *    a missing array stays empty and is rendered as "not returned".
 */

import { API_BASE, apiFetch } from './api';

export type JsonRecord = Record<string, unknown>;
export type NumberRow = Array<number | null>;
export type NumberMatrix = NumberRow[];

const DEFAULT_TIMEOUT_MS = 30_000;
/** Captures run a real forward pass (and logit lens runs one per layer). */
const COMPUTE_TIMEOUT_MS = 180_000;

/** Statuses that mean "this backend does not serve this endpoint". */
const UNSUPPORTED_STATUSES = [404, 405, 501];

export class InterpretabilityHttpError extends Error {
  readonly method: string;
  readonly path: string;
  readonly status: number;
  readonly responseText: string;

  constructor(method: string, path: string, status: number, responseText: string) {
    super(
      status === 0
        ? `${method} ${path} did not complete: ${responseText}`
        : `${method} ${path} failed: HTTP ${status}`,
    );
    this.name = 'InterpretabilityHttpError';
    this.method = method;
    this.path = path;
    this.status = status;
    this.responseText = responseText;
  }

  get isUnsupported(): boolean {
    return UNSUPPORTED_STATUSES.includes(this.status);
  }
}

export function httpStatus(error: unknown): number {
  return error instanceof InterpretabilityHttpError ? error.status : 0;
}

export function isUnsupportedError(error: unknown): boolean {
  return error instanceof InterpretabilityHttpError && error.isUnsupported;
}

/** Structured backend reason from an error body, verbatim; '' when absent. */
export function errorDetail(error: unknown): string {
  if (!(error instanceof InterpretabilityHttpError)) {
    return error instanceof Error ? error.message : '';
  }
  const parsed = tryParseJson(error.responseText);
  if (isRecord(parsed)) {
    for (const key of ['detail', 'reason', 'error', 'message']) {
      const value = parsed[key];
      if (typeof value === 'string' && value.trim()) return value.trim();
      if (Array.isArray(value)) {
        const parts = value
          .map((item) => (typeof item === 'string' ? item : JSON.stringify(item)))
          .filter((item) => item && item.trim());
        if (parts.length) return parts.join('; ');
      }
      // FastAPI's `HTTPException(detail={...})` shape, which is what
      // `backend/api/v1_contracts.py` raises for a refused dataset preview:
      // {"detail": {"code": "...", "reason": "..."}}. Flattening it would
      // lose the backend's own words, which is the whole point of showing the
      // validation error verbatim.
      if (isRecord(value)) {
        const code = asString(value.code);
        const reason = asString(value.reason) || asString(value.message) || asString(value.error);
        if (reason && code) return `${code}: ${reason}`;
        if (reason || code) return reason || code;
      }
    }
  }
  const text = error.responseText.trim();
  if (text && text.length <= 300) return text;
  return '';
}

/** Human-readable failure line: status plus the backend's own reason. */
export function describeError(error: unknown): string {
  if (!(error instanceof InterpretabilityHttpError)) {
    return error instanceof Error && error.message
      ? error.message
      : 'The request did not complete.';
  }
  const detail = errorDetail(error);
  if (error.status === 0) return detail || error.message;
  return detail ? `HTTP ${error.status}: ${detail}` : `HTTP ${error.status}`;
}

async function requestJson(
  path: string,
  init: RequestInit = {},
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<JsonRecord> {
  const method = String(init.method ?? 'GET');
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response: Response;
  try {
    response = await apiFetch(`${API_BASE}${path}`, { ...init, signal: controller.signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new InterpretabilityHttpError(method, path, 0, `timed out after ${timeoutMs} ms`);
    }
    throw new InterpretabilityHttpError(
      method,
      path,
      0,
      error instanceof Error ? error.message : String(error),
    );
  } finally {
    clearTimeout(timer);
  }

  const text = await response.text();
  if (!response.ok) {
    throw new InterpretabilityHttpError(method, path, response.status, text.slice(0, 2000));
  }
  if (!text.trim()) return {};
  const parsed = tryParseJson(text);
  if (parsed === null) {
    throw new InterpretabilityHttpError(method, path, response.status, 'response was not JSON');
  }
  // A bare array is preserved under `data` rather than being dropped.
  return isRecord(parsed) ? parsed : { data: parsed };
}

function post(path: string, body: unknown, timeoutMs?: number): Promise<JsonRecord> {
  return requestJson(
    path,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    },
    timeoutMs,
  );
}

// ---------------------------------------------------------------------------
// Endpoints (all relative to API_BASE = /api)
// ---------------------------------------------------------------------------

export function gpt2Load(modelName = 'gpt2'): Promise<JsonRecord> {
  return post('/gpt2/load', { model_name: modelName }, COMPUTE_TIMEOUT_MS);
}

export function gpt2Activations(input: { prompt: string; layer: number }): Promise<JsonRecord> {
  return post('/gpt2/activations', input, COMPUTE_TIMEOUT_MS);
}

export function gpt2LayerActivations(input: {
  prompt: string;
  layer: number;
}): Promise<JsonRecord> {
  return post('/gpt2/layer_activations', input, COMPUTE_TIMEOUT_MS);
}

/**
 * Model shape, including the layer count and the residual-stream convention.
 *
 * The capture grid needs the number of layers to know how many
 * `layer_activations` calls a capture is made of. The alternative is guessing
 * "12" from GPT-2's usual size, which would silently mislabel a different
 * checkpoint.
 */
export function gpt2Architecture(): Promise<JsonRecord> {
  return post('/gpt2/architecture', {}, COMPUTE_TIMEOUT_MS);
}

export function gpt2LogitLensAll(input: { prompt: string }): Promise<JsonRecord> {
  return post('/gpt2/logit_lens_all', input, COMPUTE_TIMEOUT_MS);
}

export function saeStatus(): Promise<JsonRecord> {
  return requestJson('/sae/status', { method: 'GET' });
}

export function saeInspect(input: {
  prompt: string;
  layer: number;
  source?: string;
  identifier?: string;
  version?: string;
}): Promise<JsonRecord> {
  return post('/sae/inspect', input, COMPUTE_TIMEOUT_MS);
}

export function listDatasets(): Promise<JsonRecord> {
  return requestJson('/datasets', { method: 'GET' });
}

export function previewDataset(input: {
  path?: string;
  dataset_id?: string;
}): Promise<JsonRecord> {
  return post('/datasets/preview', input, COMPUTE_TIMEOUT_MS);
}

// ---------------------------------------------------------------------------
// Normalizers — tolerant of shape, never inventing a value
// ---------------------------------------------------------------------------

export function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function tryParseJson(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

export function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

export function asString(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

export function asNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function asBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

/** Matrix of numbers/`null`; a non-numeric cell is `null`, never coerced to 0. */
export function numberMatrix(value: unknown): NumberMatrix {
  return asArray(value).map((row) => asArray(row).map((cell) => asNumber(cell)));
}

/** L2 norm of one returned row; `null` unless every component is a number. */
export function rowL2(row: NumberRow | null | undefined): number | null {
  if (!row || row.length === 0) return null;
  let sum = 0;
  for (const cell of row) {
    if (typeof cell !== 'number') return null;
    sum += cell * cell;
  }
  return Math.sqrt(sum);
}

export function formatNumber(value: number | null | undefined, digits = 4): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
}

export function formatInteger(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? String(Math.trunc(value)) : '—';
}

/** `[a, b, c]` → "a × b × c"; a missing shape is "not supplied". */
export function formatShape(value: unknown): string {
  const shape = asArray(value).map((item) => asNumber(item));
  if (!shape.length || shape.some((item) => item === null)) return 'not supplied';
  return shape.map((item) => formatInteger(item)).join(' × ');
}

export function provenanceOf(record: JsonRecord | null): string {
  return record ? asString(record.provenance) : '';
}

export function statusOf(record: JsonRecord | null): string {
  return record ? asString(record.status) : '';
}

/**
 * Field provenance rendered as a short line, e.g. "resid_post=live, tokens=live".
 * Returns '' when the response carries none.
 */
export function fieldProvenanceSummary(record: JsonRecord | null): string {
  if (!record || !isRecord(record.field_provenance)) return '';
  const parts = Object.entries(record.field_provenance)
    .filter(([, value]) => typeof value === 'string')
    .map(([key, value]) => `${key}=${String(value)}`);
  if (!parts.length) return '';
  const shown = parts.slice(0, 8).join(', ');
  return parts.length > 8 ? `${shown}, +${parts.length - 8} more` : shown;
}

/**
 * Token labels as returned, position-preserving.
 *
 * A non-string entry becomes '' rather than being dropped, because dropping it
 * would shift every later token onto the wrong matrix row.
 */
export function tokenLabels(record: JsonRecord | null): string[] {
  if (!record) return [];
  return asArray(record.tokens).map((token) => (typeof token === 'string' ? token : ''));
}

/** `(value ?? '—')` for a raw returned scalar. */
export function displayValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return 'Not supplied';
  if (typeof value === 'string') return value;
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : 'Not finite';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}
