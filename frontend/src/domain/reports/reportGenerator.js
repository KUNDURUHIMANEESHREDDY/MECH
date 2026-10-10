import { notebookStore } from '../notebook/notebookStore';
import { colors } from '../../design/tokens/colors';

/**
 * ReportGenerator - Generates automated experiment reports in Markdown and HTML.
 */
export class ReportGenerator {
  static generateMarkdownReport(title = 'Mechanistic Interpretability Report') {
    const notebook = notebookStore.exportNotebook();
    let md = `# ${title}\n\n`;
    md += `**Generated:** ${new Date().toISOString()}\n`;
    md += `**Provenance:** unconfirmed — this report renders notebook cells only; model identity is not asserted here.\n\n`;
    md += `## Experiment Summary & Key Observations\n`;

    let measuredCells = 0;
    notebook.cells.forEach((c) => {
      if (c.type === 'markdown') md += `${c.content}\n\n`;
      if (c.type === 'json') {
        measuredCells += 1;
        md += `\`\`\`json\n${JSON.stringify(c.data, null, 2)}\n\`\`\`\n\n`;
      }
    });

    // No hardcoded findings. The previous version asserted a Layer 8 patch
    // at L8_N402 and a Paris/France prediction delta on every report,
    // regardless of what ran. Findings are emitted only from measured cells;
    // an empty notebook yields an explicit no-measurements statement.
    md += `## Intervention & Prediction Verification\n`;
    if (measuredCells === 0) {
      md += `No measurements recorded in this notebook. No interventions were verified.\n`;
    } else {
      md += `${measuredCells} measured cell(s) recorded above; interpretation beyond the recorded values was not performed.\n`;
    }
    return md;
  }

  static generateHTMLReport(title = 'Mechanistic Interpretability Report') {
    const md = this.generateMarkdownReport(title);
    return `<!DOCTYPE html><html><head><title>${title}</title><style>body{font-family:sans-serif;padding:2rem;line-height:1.6;background:${colors.canvas};color:${colors.ink};}</style></head><body><pre>${md}</pre></body></html>`;
  }
}
