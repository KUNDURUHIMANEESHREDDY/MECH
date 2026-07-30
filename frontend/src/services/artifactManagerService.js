/**
 * Artifact Manager Service (Sprint 3 AI 5)
 * Centralized registry for reports, plots, notebooks, and exported publication figures.
 */

export class ArtifactManagerService {
  constructor() {
    this.artifacts = [
      {
        id: "art_fig1",
        title: "Circuit Graph Figure 1",
        type: "figure",
        mimeType: "image/svg+xml",
        creator: "Researcher",
        timestamp: new Date().toISOString(),
        checksum: "sha256_8a9b",
      },
      {
        id: "art_rep1",
        title: "Mechanistic Analysis Report",
        type: "report",
        mimeType: "application/json",
        creator: "Researcher",
        timestamp: new Date().toISOString(),
        checksum: "sha256_7c3f",
      },
    ];
  }

  registerArtifact(title, type, mimeType, payload) {
    const art = {
      id: `art_${Date.now().toString(36)}`,
      title,
      type,
      mimeType,
      creator: "Researcher",
      timestamp: new Date().toISOString(),
      checksum: `sha256_${Math.floor(Math.random() * 0xffff).toString(16)}`,
      payload,
    };
    this.artifacts.push(art);
    return art;
  }

  listArtifacts() {
    return [...this.artifacts];
  }
}

export const artifactManagerService = new ArtifactManagerService();
