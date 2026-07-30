/**
 * Shared Workspace Root State Manager.
 * Single root state: Workspace ➔ Session ➔ Experiment ➔ Visualization ➔ Collaboration ➔ Publication.
 */

class SharedWorkspaceState {
  constructor() {
    this.listeners = new Set();
    this.state = {
      workspaceId: 'ws_sprint4_master',
      workspaceTitle: 'GPT-2 Mechanistic Interpretability Research Workspace',
      currentSession: 'sess_default',
      currentModel: 'GPT-2 Small',
      currentLayer: 8,
      headIndex: 4,
      neuronIndex: 402,
      selectedToken: ' Paris',
      selectedFeature: 1402,
      selectedCircuit: 'c_ioi',
      selectedDiscovery: 'disc_ioi_retrieval',
      syncPlaybackStep: 2,
      hoveredElement: { type: null, id: null, layer: null },
      annotations: [
        {
          id: 'ann_1',
          target: 'L8_N402',
          author: 'Dr. Alice (Lead)',
          timestamp: '2026-07-26T10:15:00Z',
          layer: 8,
          neuronIndex: 402,
          featureId: 1402,
          confidence: 0.96,
          tags: ['Induction Head', 'IOI Circuit', 'Geographic Capital'],
          references: ['Wang et al. (2022)'],
          evidenceSnippet: 'Direct logit boost of +4.2 for Paris token upon ablation.',
          text: 'Candidate induction head active in indirect object identification.'
        },
        {
          id: 'ann_2',
          target: 'SAE #1402',
          author: 'Dr. Bob (Contributor)',
          timestamp: '2026-07-26T11:00:00Z',
          layer: 8,
          neuronIndex: 402,
          featureId: 1402,
          confidence: 0.88,
          tags: ['Polysemantic', 'Feature SAE'],
          references: ['Elhage et al. (2022)'],
          evidenceSnippet: 'Fires strongly on European capital cities and geographic nouns.',
          text: 'Polysemantic feature encoding both country names and capital relations.'
        }
      ],
      provenance: {
        experimentId: 'exp_s4_final',
        model: 'GPT-2 Small',
        checkpoint: 'sae_gpt2_l8.pt',
        paper: 'Wang et al. (2022)',
      },
      activeParticipants: [
        { id: 'user_1', name: 'Dr. Alice (Lead)', role: 'Owner', status: 'Active' },
        { id: 'user_2', name: 'Dr. Bob (Contributor)', role: 'Researcher', status: 'Viewing' }
      ]
    };
  }

  getState() {
    return { ...this.state };
  }

  updateState(patch) {
    this.state = { ...this.state, ...patch };
    this.listeners.forEach((cb) => cb(this.state));
  }

  addAnnotation(target, text, author = 'Researcher') {
    return this.addStructuredAnnotation({ target, text, author });
  }

  addStructuredAnnotation(data = {}) {
    const newAnn = {
      id: data.id || `ann_${Date.now()}`,
      target: data.target || `L${data.layer || this.state.currentLayer}_N${data.neuronIndex || this.state.neuronIndex}`,
      author: data.author || 'Researcher',
      timestamp: data.timestamp || new Date().toISOString(),
      layer: data.layer ?? this.state.currentLayer,
      neuronIndex: data.neuronIndex ?? this.state.neuronIndex,
      featureId: data.featureId ?? this.state.selectedFeature,
      confidence: data.confidence ?? 0.90,
      tags: data.tags || ['Mechanistic Annotation'],
      references: data.references || [],
      evidenceSnippet: data.evidenceSnippet || '',
      text: data.text || 'Structured research annotation.'
    };
    this.updateState({ annotations: [...this.state.annotations, newAnn] });
    return newAnn;
  }

  searchAnnotations({ tag = null, minConfidence = 0.0, layer = null } = {}) {
    return this.state.annotations.filter((ann) => {
      if (tag && !ann.tags.some((t) => t.toLowerCase().includes(tag.toLowerCase()))) return false;
      if (minConfidence && ann.confidence < minConfidence) return false;
      if (layer !== null && ann.layer !== layer) return false;
      return true;
    });
  }

  setHoveredElement(type, id, layer = null) {
    this.updateState({ hoveredElement: { type, id, layer } });
  }

  clearHoveredElement() {
    this.updateState({ hoveredElement: { type: null, id: null, layer: null } });
  }

  subscribe(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }
}

export const sharedWorkspaceState = new SharedWorkspaceState();
