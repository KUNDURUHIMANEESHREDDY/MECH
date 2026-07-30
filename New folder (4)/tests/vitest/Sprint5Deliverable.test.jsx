import { describe, it, expect } from 'vitest';
import { aiScientistEcosystem } from '../../src/services/aiScientistEcosystem';

describe('AI Scientist Ecosystem Sprint 5 Suite', () => {
  it('initializes all 11 core ecosystem services correctly', () => {
    const summary = aiScientistEcosystem.getEcosystemSummary();
    expect(summary.status).toBe('Active');
    expect(summary.services.length).toBe(11);
    expect(summary.activeProjectsCount).toBe(3);
    expect(summary.installedPluginsCount).toBe(2);
    expect(summary.registryCatalogCount).toBe(5);
    expect(summary.templatesCount).toBe(4);
  });

  it('runs reproducibility verification and package publishing pipeline', () => {
    const pkg = aiScientistEcosystem.packages.createPackage('pkg_sae_probe', 'SAE Feature Probing Package');
    expect(pkg.status).toBe('Published');

    const repro = aiScientistEcosystem.reproducibility.verifyReproducibility('disc_ioi_retrieval');
    expect(repro.verified).toBe(true);
    expect(repro.score).toBeGreaterThan(0.9);

    const lineage = aiScientistEcosystem.provenance.getLineage('disc_ioi_retrieval');
    expect(lineage.verifiedLineage).toBe(true);
    expect(lineage.lineage.length).toBe(6);

    const deps = aiScientistEcosystem.dependencyGraph.getDependencies('pkg_ioi_circuit');
    expect(deps.totalDependenciesCount).toBe(2);

    const tmpl = aiScientistEcosystem.templates.listTemplates();
    expect(tmpl.length).toBe(4);
  });
});
