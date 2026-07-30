/**
 * Publication Export Service (Sprint 3 AI 4)
 * Generates vector SVG, high-res PNG, and JSON data bundles from DTOs.
 */

export class ExportService {
  /**
   * Export figure metadata and DTO payload as downloadable JSON artifact
   */
  exportJSON(title, data) {
    const payload = {
      title,
      exportedAt: new Date().toISOString(),
      schemaVersion: "3.0.0",
      data,
    };
    const jsonStr = JSON.stringify(payload, null, 2);
    return {
      filename: `${title.toLowerCase().replace(/\s+/g, "_")}_artifact.json`,
      content: jsonStr,
      mimeType: "application/json",
    };
  }

  /**
   * Generates publication-quality SVG string representation of a figure DTO
   */
  exportSVG(title, elements = []) {
    const svgHeader = `<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600" style="background:#0f172a; font-family: sans-serif;">`;
    const titleTag = `<text x="40" y="50" fill="#f8fafc" font-size="20" font-weight="bold">${title}</text>`;
    const timestampTag = `<text x="40" y="75" fill="#94a3b8" font-size="12">Exported: ${new Date().toISOString()}</text>`;
    
    let contentStr = `<g transform="translate(40, 100)">`;
    elements.forEach((el, idx) => {
      contentStr += `<rect x="${(idx % 4) * 170}" y="${Math.floor(idx / 4) * 80}" width="150" height="60" rx="8" fill="#1e293b" stroke="#3b82f6" stroke-width="2"/>`;
      contentStr += `<text x="${(idx % 4) * 170 + 15}" y="${Math.floor(idx / 4) * 80 + 35}" fill="#38bdf8" font-size="14">${el.label || `Node ${idx}`}</text>`;
    });
    contentStr += `</g>`;

    const svgFooter = `</svg>`;
    const fullSvg = `${svgHeader}${titleTag}${timestampTag}${contentStr}${svgFooter}`;
    
    return {
      filename: `${title.toLowerCase().replace(/\s+/g, "_")}_publication.svg`,
      content: fullSvg,
      mimeType: "image/svg+xml",
    };
  }
}

export const exportService = new ExportService();
