import { useState, useEffect } from 'react';
import type { SocietyEvent } from '../services/societyService';

export type SocietyPhase = 'idle' | 'running' | 'done' | 'failed';

export interface SocietyTraceStep {
  node: string;
  agent: string;
  status: string;
  error?: string;
  reason?: string;
}

export interface SocietyState {
  goal: string;
  phase: SocietyPhase;
  runId: string | null;
  events: SocietyEvent[];
  steps: SocietyTraceStep[];
  summary: string;
  reportMd: string;
  error: string;
}

const DEFAULT_SOCIETY_STATE: SocietyState = {
  goal: 'Reproduce IOI on gpt2-small, find causally important heads',
  phase: 'idle',
  runId: null,
  events: [],
  steps: [],
  summary: '',
  reportMd: '',
  error: '',
};

let globalSocietyState: SocietyState = DEFAULT_SOCIETY_STATE;
const listeners = new Set<() => void>();

export const societyStore = {
  getState: () => globalSocietyState,
  setState: (fn: ((prev: SocietyState) => Partial<SocietyState>) | Partial<SocietyState>) => {
    const patch = typeof fn === 'function' ? fn(globalSocietyState) : fn;
    globalSocietyState = { ...globalSocietyState, ...patch };
    listeners.forEach(l => l());
  },
  subscribe: (listener: () => void) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
};

export function useSocietyStore(): [SocietyState, typeof societyStore.setState] {
  const [state, setState] = useState<SocietyState>(globalSocietyState);

  useEffect(() => {
    return societyStore.subscribe(() => setState(globalSocietyState));
  }, []);

  return [state, societyStore.setState];
}
