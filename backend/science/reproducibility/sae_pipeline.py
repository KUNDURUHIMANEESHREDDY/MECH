"""SAE Reproduction Pipeline.

Would reproduce Bricken et al. 2023 — "Towards Monosemanticity: Decomposing
Language Models With Dictionary Learning" — by training a sparse autoencoder on
GPT-2 MLP post-activation residuals and measuring L0 sparsity, reconstruction
error and feature monosemanticity.

Status: NOT IMPLEMENTED
-----------------------
This module previously drew its entire feature bank from a seeded RNG::

    act_freq = round(rng.betavariate(0.5, 5.0), 4)
    top_tokens = rng.sample(token_pool, min(n_top, len(token_pool)))
    mono = round(rng.betavariate(3.0, 1.5), 4)
    is_absorbed = rng.random() < 0.12

and reported them under field names that read as findings: `top_activating_tokens`
was a random sample from a 19-word list, `monosemanticity_score` was a betavariate
draw, and `feature_absorption_rate` was a coin flip at p=0.12.

Three things made this worse than the other simulated pipelines:

* **It ignored `mock_mode`.** `_simulate_sae_features` was called unconditionally,
  so a caller with real GPT-2 weights loaded still received simulated features.
  The flag on the constructor was decorative. Contrast the arithmetic pipeline,
  whose fabrication at least was unconditional *and* obvious.
* **`reconstruction_mse` was not a reconstruction.** It was
  `0.03 + (1 - mean_monosemanticity) * 0.04` -- a function of the simulated
  scores. No encoding or decoding ran, so the number had no relationship to any
  autoencoder.
* **The provenance was actively false.** `dataset_name="OpenWebText Sample"` was
  written into a dataset manifest, so every report claimed a corpus that this
  platform never read. The `explanation_of_diffs` did disclose the simulation,
  which is to its credit, but disclosure in a metadata field is not a substitute
  for not fabricating: the disclosure is dropped by every consumer that reads
  only `observed_metrics`.

The numbers are removed rather than gated behind a mock flag. A conditional stub
differs from an unconditional one only in which caller receives the fiction.

What implementing this actually requires:
  1. Activations: collect real MLP post-activation residuals from loaded GPT-2
     weights over a real corpus. Record the corpus, its version and its licence.
  2. Training: fit a top-k or JumpReLU SAE to those activations. Report
     reconstruction error from an actual encode/decode round trip.
  3. Evaluation: compute L0 from the real reconstruction, and score
     monosemanticity with a stated method (e.g. max-activation decoding against
     a held-out token sample) rather than a distribution chosen to look
     plausible.

A cheaper honest alternative exists and is deliberately *not* implemented here:
report raw MLP activation sparsity on real weights. That is a real, correctly
labelled quantity, but it is not SAE reconstruction error and must not be
reported under these field names. Whoever does it should rename the fields rather
than reuse this contract.
"""

from __future__ import annotations

from typing import Any

from ..models.adapter_base import LiveUnavailable


class SAEReproductionPipeline:
    """Not implemented. Raises rather than reporting simulated features."""

    PAPER_ID = "sparse_autoencoders"

    def __init__(self, mock_mode: bool = False) -> None:
        # Default False, matching GPT2Adapter and GreaterThanCircuitPipeline.
        # The previous default of True, combined with a simulation that ignored
        # it, meant the flag controlled nothing.
        self.mock_mode = mock_mode

    def run(self, n_features: int = 50, seed: int = 42) -> Dict[str, Any]:
        raise LiveUnavailable(
            "SAEReproductionPipeline is not implemented. It previously returned "
            "50 features whose top_activating_tokens were drawn at random from a "
            "19-word list, whose monosemanticity_score was a betavariate draw, "
            "and whose reconstruction_mse was 0.03 + (1 - mean_score) * 0.04 -- "
            "a formula over those same random numbers, with no autoencoder "
            "trained or evaluated. The simulation ran regardless of mock_mode, "
            "and the dataset manifest claimed OpenWebText, which was never "
            "read. No SAE is trained by this platform. See the module "
            "docstring for what a real implementation requires."
        )
