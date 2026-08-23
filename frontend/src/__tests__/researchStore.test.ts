import { describe, it, expect, beforeEach } from 'vitest';
import { useResearchStore } from '../shared/stores/research';

describe('useResearchStore', () => {
  beforeEach(() => {
    useResearchStore.setState({
      investigations: [],
      activeInvestigationId: null,
      activeInvestigation: null,
      hypotheses: [],
      activeHypothesisId: null,
      activeHypothesis: null,
      runs: [],
      evidence: [],
      evidenceMatrix: [],
      mechanisms: [],
      selectedComponent: null,
      selectedTokenIndex: null,
      isDemoMode: false,
    });
  });

  it('initializes with default values', () => {
    const state = useResearchStore.getState();
    expect(state.investigations).toEqual([]);
    expect(state.activeInvestigation).toBeNull();
    expect(state.isDemoMode).toBe(false);
  });

  it('selects and updates component coordinate synchronously', () => {
    const store = useResearchStore.getState();
    store.selectComponent({ name: 'L9H9', layer: 9, head: 9, componentType: 'head' });

    const updated = useResearchStore.getState();
    expect(updated.selectedComponent).toEqual({
      name: 'L9H9',
      layer: 9,
      head: 9,
      componentType: 'head',
    });
  });

  it('selects and updates token index synchronously', () => {
    const store = useResearchStore.getState();
    store.selectToken(4);

    const updated = useResearchStore.getState();
    expect(updated.selectedTokenIndex).toBe(4);
  });
});
