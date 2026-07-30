/**
 * Dataset Versioning Service.
 */

export class DatasetVersioningService {
  constructor() {
    this.datasets = [
      { id: 'ds_1', name: 'IOI Prompts Dataset', version: 'v1.2.0', checksum: 'sha256_a402f891', samples: 1000 },
      { id: 'ds_2', name: 'OpenWebText Interpretability Slice', version: 'v2.0.0', checksum: 'sha256_b8192c73', samples: 50000 },
    ];
  }

  listDatasets() {
    return [...this.datasets];
  }

  listDatasetVersions(datasetId = 'ds_1') {
    return [
      { datasetId, version: 'v1.0.0', createdAt: '2026-01-01T00:00:00Z' },
      { datasetId, version: 'v1.2.0', createdAt: '2026-03-15T00:00:00Z' },
    ];
  }

  createVersion(datasetId, newVersion) {
    return {
      datasetId,
      newVersion,
      checksum: `sha256_${Date.now()}`,
      createdAt: new Date().toISOString(),
    };
  }
}

export const datasetVersioning = new DatasetVersioningService();
