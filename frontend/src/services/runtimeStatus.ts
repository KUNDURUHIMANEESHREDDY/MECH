/**
 * Runtime status vocabulary for the app shell.
 *
 * Four states, because "connected" used to mean two different claims:
 *
 * - connecting    a probe is in flight
 * - ready         the API answered. This is a claim about the control plane
 *                 ONLY — no model weights were executed, so the shell must
 *                 never say "N models available" from a catalogue response.
 * - disconnected  the API stopped answering after it had answered
 * - failed        the API never answered, or refused the request (auth/
 *                 permission). Distinct from a dropped connection.
 *
 * `GET /api/models` is not the readiness probe: it returns a static reference
 * catalogue with `provenance: "reference"`, which is true even when no weights
 * are cached. Probing `/api/status` and describing the API keeps the pill from
 * advertising research capability that does not exist.
 */

export type RuntimeState = 'connecting' | 'ready' | 'disconnected' | 'failed';

export interface RuntimeStatus {
  state: RuntimeState;
  /** Short text rendered in the sidebar pill. */
  label: string;
  /** Full sentence for the tooltip / assistive technology. */
  detail: string;
  /** Whether a user-initiated retry makes sense right now. */
  retryable: boolean;
}

export interface ProbeResult {
  ok: boolean;
  /** HTTP status when the API answered with an error. */
  status?: number;
  /** Transport/HTTP error message when `ok` is false. */
  message?: string;
}

export function statusFor(
  state: RuntimeState,
  label: string,
  detail: string,
): RuntimeStatus {
  return {
    state,
    label,
    detail,
    retryable: state === 'disconnected' || state === 'failed',
  };
}

export function connectingStatus(): RuntimeStatus {
  return statusFor(
    'connecting',
    'Connecting to backend…',
    'Probing GET /api/status. No research result is shown until the backend answers.',
  );
}

export function readyStatus(): RuntimeStatus {
  return statusFor(
    'ready',
    'Backend ready',
    'GET /api/status answered. Model weights are not implied: load a model before running research tools.',
  );
}

/**
 * Classify one probe outcome. `previous` is the state the shell was in when
 * the probe started, so a connection that drops is reported as a
 * disconnection rather than a backend that never existed.
 */
export function classifyProbe(
  result: ProbeResult,
  previous: RuntimeState,
): RuntimeStatus {
  if (result.ok) return readyStatus();

  if (result.status === 401 || result.status === 403) {
    return statusFor(
      'failed',
      'Unauthorized — backend rejected the token',
      `GET /api/status answered HTTP ${result.status}. The backend is reachable but refused this token; research calls will fail until a valid bearer token is configured.`,
    );
  }

  const message = result.message?.trim() || 'No response from GET /api/status.';

  if (previous === 'ready') {
    return statusFor(
      'disconnected',
      'Disconnected — backend unreachable',
      `${message} The backend answered earlier in this session, so results already on screen are retained as stale rather than dropped.`,
    );
  }

  return statusFor(
    'failed',
    'Backend failed — no response',
    `${message} Nothing has answered yet, so no model or dataset is available.`,
  );
}
