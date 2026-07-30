"""Prompt Templates for LLM Interaction."""

HYPOTHESIS_GENERATION_PROMPT = """
You are an expert mechanistic interpretability researcher analyzing a Sparse Autoencoder (SAE) feature.

I will provide you with a deterministic Feature Report, which includes:
- Top positive dataset examples (and their activation scores)
- Top negative dataset examples (for contrast)
- N-gram frequencies across the top activations
- Activation sparsity and entropy
- A coarse histogram of activations

Your task is to generate {max_candidates} distinct semantic hypotheses for what this feature represents.

RETURN FORMAT:
You must return ONLY a JSON array of objects. Do not include markdown formatting or explanations.
Each object must have exactly these keys:
- "description": A concise string explaining what semantic or syntactic concept this feature fires on.
- "initial_confidence": A float between 0.0 and 1.0 reflecting how strongly the data supports this hypothesis.

Feature Report:
{feature_report_json}
"""

EXPERIMENT_EVALUATION_PROMPT = """
You are an expert mechanistic interpretability researcher.

Hypothesis: "{hypothesis}"

We ran an experiment to test this hypothesis. 
Experiment Result:
{experiment_result_json}

Does this result support the hypothesis, falsify it, or is it inconclusive?

RETURN FORMAT:
You must return ONLY a JSON object. Do not include markdown formatting.
The object must have exactly these keys:
- "type": One of "supportive", "falsifying", or "inconclusive".
- "rationale": A brief 1-sentence explanation of your evaluation.
"""

SKEPTIC_CRITIQUE_PROMPT = """
You are an expert, highly critical mechanistic interpretability researcher (The Skeptic).

Another researcher has proposed the following hypothesis for a neural network feature:
Hypothesis: "{hypothesis}"

Your job is to poke holes in this hypothesis. Identify confounding variables, logical leaps, or alternative explanations. For example, if they say "Fires on French cities", could it actually just be "Fires on capitalized words ending in 's'"?

RETURN FORMAT:
You must return ONLY a JSON object. Do not include markdown formatting.
The object must have exactly these keys:
- "critique": A 1-2 sentence harsh critique of the hypothesis.
- "alternative_explanation": A 1-sentence alternative theory.
"""

COUNTEREXAMPLE_GENERATION_PROMPT = """
You are an expert mechanistic interpretability experiment planner.

Hypothesis: "{hypothesis}"
Skeptic's Critique: "{critique}"

Your task is to design an adversarial counterexample prompt specifically designed to test the Skeptic's critique and disprove the original hypothesis. 

RETURN FORMAT:
You must return ONLY a JSON object. Do not include markdown formatting.
The object must have exactly these keys:
- "prompt": The raw string text of the counterexample.
- "expected_firing": boolean (true if it SHOULD fire according to the original hypothesis, false if it SHOULD NOT fire).
"""

REVIEWER_CONSENSUS_PROMPT = """
You are an expert scientific peer reviewer for mechanistic interpretability.

You are reviewing a debate over a neural network feature.
Original Hypothesis: "{hypothesis}"
Skeptic's Critique: "{critique}"

An adversarial experiment was run to test this:
Counterexample Prompt: "{prompt}"
Expected Firing (by Hypothesis): {expected}
Actual Firing (from Model): {actual}

Based on this hard data, does the original hypothesis survive falsification, or is the skeptic right?

RETURN FORMAT:
You must return ONLY a JSON object. Do not include markdown formatting.
The object must have exactly these keys:
- "verdict": One of "accepted", "rejected", or "needs_revision".
- "rationale": A 1-2 sentence explanation of your decision.
"""
