import { eventBus } from '../../utils/eventBus';

/**
 * NotebookStore - Manages experiment research notebook cells (Markdown, Plot, Image, Table, JSON, Code).
 */
class NotebookStore {
  constructor() {
    // A fresh notebook starts EMPTY. Seed cells used to carry hardcoded
    // "findings" (residual norms, L8_N402 scores, feature 1402 @ 4.12) that
    // flowed verbatim into exported reports — invented measurements with a
    // fresh timestamp. Cells below are an explicitly-labeled template with
    // no numeric claims; real cells are appended only from backend runs.
    this.cells = [
      { id: 'c1', type: 'markdown', content: '# Experiment Notebook\nEmpty template — cells added here come only from executed backend runs. Nothing below is a measurement.' },
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
