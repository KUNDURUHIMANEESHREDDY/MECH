# Scientific Methods

This document explains *why* each interpretability method exists in the platform.

## Activation Patching
**Why:** To determine causal importance. By intervening on activations during a forward pass and measuring the effect on a metric (e.g., logit difference), we establish causality rather than mere correlation.

## Path Patching
**Why:** To isolate specific computational edges. Standard activation patching affects all downstream nodes. Path patching restricts the intervention to a specific path (e.g., Head A -> Head B), allowing for precise circuit discovery.

## Sparse Autoencoders (SAEs)
**Why:** To resolve superposition. Neural networks represent many concepts using fewer dimensions (superposition). SAEs decompose these dense, polysemantic activations into sparse, monosemantic features that humans can understand.

## Logit Lens
**Why:** To observe concept formation across depth. Projecting intermediate hidden states to the vocabulary space reveals when the model "decides" on its final prediction.
