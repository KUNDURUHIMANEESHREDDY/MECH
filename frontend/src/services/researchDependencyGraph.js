export const researchDependencyGraph = {
  getDependencies(packageId = 'pkg_ioi_circuit') {
    const map = {
      pkg_ioi_circuit: ['ds_ioi_prompts', 'pkg_sae_features'],
      pkg_sae_features: ['mdl_gpt2_weights'],
      pkg_publication_paper: ['pkg_ioi_circuit', 'bench_ioi_suite'],
    };
    const deps = map[packageId] || [];
    return { packageId, directDependencies: deps, totalDependenciesCount: deps.length };
  },
};
