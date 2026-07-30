/**
 * Laboratory Management Service (AI 5).
 */

export class LabManagementService {
  constructor() {
    this.organizations = [
      { id: 'org_1', name: 'AI Safety & Interpretability Lab', computeBudgetUsd: 50000, storageQuotaTb: 10 },
    ];
    this.projects = [
      { id: 'proj_1', name: 'IOI Circuit Analysis' },
      { id: 'proj_2', name: 'Induction Head Detection' },
      { id: 'proj_3', name: 'Superposition Analysis' },
    ];
  }

  listProjects() {
    return [...this.projects];
  }

  getLabOverview() {
    return {
      organizations: this.organizations,
      projects: this.projects,
      totalUsers: 12,
      activeJobs: 4,
      monthlyComputeSpentUsd: 1420.5,
    };
  }
}

export const labManagementService = new LabManagementService();
