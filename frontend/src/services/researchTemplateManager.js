/**
 * Research Template Manager (Sprint 3 AI 5)
 * Preset research templates for SAE Analysis, Circuit Discovery, and Logit Lens.
 */

export class ResearchTemplateManager {
  listTemplates() {
    return [
      {
        id: "template_circuit_discovery",
        name: "Automated Circuit Discovery",
        description: "Extract computational circuit graph Neuron -> SAE Feature -> Head -> Output.",
        steps: ["Load Model", "Run Causal Tracing", "Extract Circuit", "Generate Report"],
      },
      {
        id: "template_sae_analysis",
        name: "Sparse Autoencoder Feature Probe",
        description: "Inspect 16k SAE features over clean/corrupted prompts.",
        steps: ["Load SAE Checkpoint", "Filter Firing Features", "Cluster Features"],
      },
      {
        id: "template_logit_lens",
        name: "Layer Logit Projection",
        description: "Project intermediate residual stream layers into unembedding vocabulary space.",
        steps: ["Execute Forward Pass", "Apply Logit Lens", "Apply Tuned Lens"],
      },
    ];
  }

  instantiateTemplate(templateId, prompt = "The capital of France is") {
    const tpl = this.listTemplates().find((t) => t.id === templateId);
    if (!tpl) throw new Error(`Template ${templateId} not found`);
    return {
      instanceId: `inst_${Date.now().toString(36)}`,
      template: tpl,
      prompt,
      status: "initialized",
    };
  }
}

export const researchTemplateManager = new ResearchTemplateManager();
