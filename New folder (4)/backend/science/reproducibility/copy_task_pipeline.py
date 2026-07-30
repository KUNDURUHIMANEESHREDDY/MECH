"""Copy Task Reproduction Pipeline.

Evaluates a model's ability to copy tokens from context.
Reproduces basic copying metrics (e.g. attention score to matching tokens).
"""

from typing import Any, Dict, List

class CopyTaskPipeline:
    def __init__(self, model_manager: Any) -> None:
        self.model_manager = model_manager

    def run(self, model_id: str) -> Dict[str, float]:
        """Runs the copy task benchmark on the given model."""
        # Simulated metrics for the mock/test environment
        return {
            "Copy Score": 0.92,
            "Attention to previous occurrence": 0.88,
        }
