/**
 * Experiment Marketplace Service.
 */

export class ExperimentMarketplace {
  constructor() {
    this.experiments = [
      { id: 'exp_mkt_1', name: 'IOI Circuit Discovery Template', author: 'Community', rating: 4.9, downloads: 1420 },
      { id: 'exp_mkt_2', name: 'SAE Polysemanticity Suite', author: 'AI Safety Lab', rating: 4.8, downloads: 890 }
    ];
  }

  listMarketplace() {
    return [...this.experiments];
  }
}
