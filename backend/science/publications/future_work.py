from typing import Dict, Any, List

class FutureWorkGenerator:
    """
    Generates realistic future research directions based on current experiment results.
    """
    
    @staticmethod
    def generate_directions(results: Dict[str, Any]) -> str:
        model = results.get("model", "the target model")
        is_reproduced = results.get("reproduction_successful", False)
        
        directions = [
            f"Extend the current analysis to larger models within the same family (e.g., scaling from {model} to its larger variants).",
            "Perform cross-family comparisons (e.g., Llama-3 vs Gemma) to evaluate the universality of the discovered circuits.",
            "Investigate the emergence of these features during training by analyzing intermediate checkpoints.",
            "Apply automated circuit discovery (ACDC) to expand the scope of the identified subnetwork."
        ]
        
        if not is_reproduced:
            directions.append("Investigate the discrepancies between this reproduction and original published results, focusing on dataset drift or model versioning.")
            
        return "## Future Work\n\n" + "\n".join([f"* {d}" for d in directions])
