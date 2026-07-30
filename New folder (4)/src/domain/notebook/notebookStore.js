import { eventBus } from '../../utils/eventBus';

/**
 * NotebookStore - Manages experiment research notebook cells (Markdown, Plot, Image, Table, JSON, Code).
 */
class NotebookStore {
  constructor() {
    this.cells = [
      { id: 'c1', type: 'markdown', content: '# GPT-2 Indirect Object Identification Experiment\nInvestigating layer 8 feature activations on IOI sequences.' },
      { id: 'c2', type: 'plot', title: 'Layer 8 Residual Norms', data: [12.4, 14.1, 16.5, 18.2] },
      { id: 'c3', type: 'table', headers: ['Layer', 'Head', 'Score'], rows: [[8, 9, 0.95], [9, 9, 0.88]] },
      { id: 'c4', type: 'json', data: { feature_id: 1402, activation: 4.12, label: 'Indirect Object' } },
    ];
  }

  getCells() {
    return [...this.cells];
  }

  addCell(type = 'markdown', content = '') {
    const newCell = { id: `c_${Date.now()}`, type, content, timestamp: new Date().toISOString() };
    this.cells.push(newCell);
    eventBus.emit('notebook:updated', this.getCells());
    return newCell;
  }

  updateCell(id, content) {
    const cell = this.cells.find((c) => c.id === id);
    if (cell) {
      cell.content = content;
      eventBus.emit('notebook:updated', this.getCells());
    }
  }

  exportNotebook() {
    return {
      version: '1.0',
      title: 'Mechanistic Interpretability Experiment Notebook',
      cells: this.getCells(),
      created_at: new Date().toISOString(),
    };
  }
}

export const notebookStore = new NotebookStore();
