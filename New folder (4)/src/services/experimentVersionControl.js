/**
 * Experiment Version Control (Sprint 3 AI 5)
 * Manages experiment DAG history commits, snapshots, and rollback functionality.
 */

export class ExperimentVersionControl {
  constructor() {
    this.commits = [
      {
        id: "commit_init",
        message: "Initial model run",
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        author: "Researcher",
      },
      {
        id: "commit_circuit",
        message: "Discovered IOI circuit",
        timestamp: new Date(Date.now() - 1800000).toISOString(),
        author: "Researcher",
      },
    ];
  }

  commit(message, state = {}) {
    const commitObj = {
      id: `commit_${Date.now().toString(36)}`,
      message,
      timestamp: new Date().toISOString(),
      author: "Researcher",
      stateSnapshot: state,
    };
    this.commits.push(commitObj);
    return commitObj;
  }

  getHistory() {
    return [...this.commits];
  }

  rollback(commitId) {
    const target = this.commits.find((c) => c.id === commitId);
    if (!target) throw new Error(`Commit ${commitId} not found`);
    return { status: "rolled_back", commit: target };
  }
}

export const experimentVersionControl = new ExperimentVersionControl();
