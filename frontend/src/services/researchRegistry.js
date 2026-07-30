export const researchRegistry = {
  listCatalog(itemType = null) {
    return [
      { id: 'exp_1', type: 'Experiment', name: 'IOI Circuit Tracing', version: '1.0.0' },
      { id: 'disc_1', type: 'Discovery', name: 'Layer 8 Induction Head', version: '1.2.0' },
      { id: 'bench_1', type: 'Benchmark', name: 'IOI Benchmark Suite', version: '2.0.0' },
      { id: 'ds_1', type: 'Dataset', name: 'IOI Prompts v1', version: '1.0.0' },
      { id: 'pub_1', type: 'Publication', name: 'IOI Paper Package', version: '1.0.0' },
    ].filter((i) => !itemType || i.type.toLowerCase() === itemType.toLowerCase());
  },
};
