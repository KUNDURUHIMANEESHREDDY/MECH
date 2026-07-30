/**
 * Central Collaboration Engine (Sprint 4).
 * Coordinates accounts, permissions, real-time sync, peer review, experiment marketplace, dataset hub, plugin marketplace, and publication pipeline.
 */

import { UserAccountService } from './userAccountService.js';
import { RealtimeCollabEngine } from './realtimeCollabEngine.js';
import { ReviewSystem } from './reviewSystem.js';
import { ExperimentMarketplace } from './experimentMarketplace.js';
import { DatasetHub } from './datasetHub.js';
import { PluginMarketplaceV2 } from './pluginMarketplaceV2.js';
import { ArXivPublicationPipeline } from './arxivPublicationPipeline.js';

class CollaborationEngine {
  constructor() {
    this.accountService = new UserAccountService();
    this.collabEngine = new RealtimeCollabEngine();
    this.reviewSystem = new ReviewSystem();
    this.experimentMarketplace = new ExperimentMarketplace();
    this.datasetHub = new DatasetHub();
    this.pluginMarketplace = new PluginMarketplaceV2();
    this.publicationPipeline = new ArXivPublicationPipeline();
  }

  getWorkspaceOverview() {
    return {
      users: this.accountService.listUsers(),
      collabHistory: this.collabEngine.getHistory(),
      reviews: this.reviewSystem.listReviews(),
      experiments: this.experimentMarketplace.listMarketplace(),
      datasets: this.datasetHub.listDatasets(),
      plugins: this.pluginMarketplace.listPlugins(),
    };
  }
}

export const collaborationEngine = new CollaborationEngine();
