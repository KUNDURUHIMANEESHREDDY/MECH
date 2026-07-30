export const provenanceViewer = {
  getLineage(discoveryId = 'disc_s5_master') {
    return {
      discoveryId,
      lineage: [
        { step: 1, stage: 'Dataset', name: 'IOI Prompts v1', id: 'ds_ioi_1' },
        { step: 2, stage: 'Model', name: 'GPT-2 Small', id: 'mdl_gpt2' },
        { step: 3, stage: 'Prompt', name: 'When John and Mary...', id: 'prm_102' },
        { step: 4, stage: 'Runtime', name: 'Ray Execution Cluster', id: 'rt_ray_4' },
        { step: 5, stage: 'Discovery', name: 'L8_N402 IOI Circuit', id: discoveryId },
        { step: 6, stage: 'Paper', name: 'ArXiv Package #9402', id: 'paper_9402' },
      ],
      verifiedLineage: true,
    };
  },
};
