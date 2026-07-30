/**
 * Base ResearchArtifact Schema.
 * Standardized schema inherited by all research artifacts (Notebooks, Figures, Reports, Datasets, Circuits, Discoveries, Papers).
 */

export class ResearchArtifact {
  constructor({
    id,
    type,
    title,
    version = '1.0.0',
    creator = 'Dr. Mechanistic Researcher',
    timestamp = new Date().toISOString(),
    provenance = { prompt: 'The capital of France is', model: 'GPT-2 Small' },
    dependencies = [],
    metadata = {}
  }) {
    this.id = id;
    this.type = type;
    this.title = title;
    this.version = version;
    this.creator = creator;
    this.timestamp = timestamp;
    this.provenance = provenance;
    this.dependencies = dependencies;
    this.metadata = metadata;
  }

  toJSON() {
    return {
      id: this.id,
      type: this.type,
      title: this.title,
      version: this.version,
      creator: this.creator,
      timestamp: this.timestamp,
      provenance: this.provenance,
      dependencies: this.dependencies,
      metadata: this.metadata,
    };
  }
}
