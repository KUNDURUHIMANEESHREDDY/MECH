export interface ResearchSelectionContext {
  feature?: number;
  layer?: number;
  neuron?: number;
  token?: number;
  prompt?: string;
  checkpoint?: string;
  source?: string;
  version?: string;
  origin?: string;
}
const KEY = 'mech:research-selection:v1';
export function setResearchSelection(context: ResearchSelectionContext): void {
  try { window.sessionStorage.setItem(KEY, JSON.stringify(context)); } catch { /* storage may be disabled */ }
}
export function takeResearchSelection(): ResearchSelectionContext | null {
  try {
    const raw = window.sessionStorage.getItem(KEY);
    if (!raw) return null;
    window.sessionStorage.removeItem(KEY);
    const parsed: unknown = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed as ResearchSelectionContext : null;
  } catch { return null; }
}
