/**
 * useBackendStatus.ts
 *
 * Polls the MECH FastAPI /health endpoint every 5 seconds and exposes
 * whether the Python sidecar is alive.  All UI provenance badges should
 * subscribe to this hook rather than doing ad-hoc fetch calls.
 */
import { useState, useEffect, useCallback } from 'react';

export type BackendStatus = 'LIVE_PYTORCH' | 'BACKEND_OFFLINE' | 'CHECKING';

const HEALTH_URL = 'http://localhost:8000/health';
const POLL_INTERVAL_MS = 5000;

export function useBackendStatus(): BackendStatus {
  const [status, setStatus] = useState<BackendStatus>('CHECKING');

  const check = useCallback(async () => {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2500);
      const res = await fetch(HEALTH_URL, {
        method: 'GET',
        signal: controller.signal,
      });
      clearTimeout(timeout);
      setStatus(res.ok ? 'LIVE_PYTORCH' : 'BACKEND_OFFLINE');
    } catch {
      setStatus('BACKEND_OFFLINE');
    }
  }, []);

  useEffect(() => {
    check();
    const interval = setInterval(check, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [check]);

  return status;
}
