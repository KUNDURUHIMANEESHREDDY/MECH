/**
 * Dataset Hub Service.
 */

export class DatasetHub {
  constructor() {
    this.datasets = [
      { id: 'ds_openwebtext', name: 'OpenWebText Sample', version: '2.1.0', samplesCount: 50000 },
      { id: 'ds_ioi_prompts', name: 'Indirect Object Identification Prompts', version: '1.0.0', samplesCount: 1000 }
    ];
  }

  listDatasets() {
    return [...this.datasets];
  }
}
