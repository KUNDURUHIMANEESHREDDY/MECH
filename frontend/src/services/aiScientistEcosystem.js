/**
 * Integrated AI Scientist Ecosystem Services (Sprint 5 Infrastructure Suite).
 * Connects distributed collaboration, package manager, research CI, dataset versioning,
 * plugin registry, reproducibility verifier, lab management, research registry,
 * provenance viewer, dependency graph, and experiment templates.
 */

import { distributedCollaboration } from './distributedCollaboration';
import { researchPackageManager } from './researchPackageManager';
import { researchCIEngine } from './researchCIEngine';
import { datasetVersioning } from './datasetVersioning';
import { pluginRegistryService } from './pluginRegistryService';
import { reproducibilityVerifier } from './reproducibilityVerifier';
import { labManagementService } from './labManagementService';

import { researchRegistry } from './researchRegistry';
import { provenanceViewer } from './provenanceViewer';
import { researchDependencyGraph } from './researchDependencyGraph';
import { experimentTemplates } from './experimentTemplates';

export const aiScientistEcosystem = {
  collaboration: distributedCollaboration,
  packages: researchPackageManager,
  ci: researchCIEngine,
  datasets: datasetVersioning,
  plugins: pluginRegistryService,
  reproducibility: reproducibilityVerifier,
  lab: labManagementService,

  registry: researchRegistry,
  provenance: provenanceViewer,
  dependencyGraph: researchDependencyGraph,
  templates: experimentTemplates,

  getEcosystemSummary() {
    return {
      status: 'Active',
      version: '5.0.0',
      services: [
        'DistributedCollaboration',
        'ResearchPackageManager',
        'ResearchCIEngine',
        'DatasetVersioning',
        'PluginRegistryService',
        'ReproducibilityVerifier',
        'LabManagementService',
        'ResearchRegistry',
        'ProvenanceViewer',
        'ResearchDependencyGraph',
        'ExperimentTemplates',
      ],
      activeProjectsCount: this.lab.listProjects().length,
      installedPluginsCount: this.plugins.listPlugins().length,
      datasetVersionsCount: this.datasets.listDatasetVersions('ds_ioi_prompts').length,
      registryCatalogCount: this.registry.listCatalog().length,
      templatesCount: this.templates.listTemplates().length,
    };
  },
};
