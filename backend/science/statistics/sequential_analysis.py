from typing import Dict, Any, List

class SequentialAnalysis:
    """
    Evaluates whether enough evidence exists to stop an experiment early.
    """

    @staticmethod
    def evaluate(
        current_n: int, 
        current_p_value: float, 
        current_power: float, 
        target_power: float = 0.8,
        alpha: float = 0.05,
        min_n: int = 30
    ) -> Dict[str, Any]:
        """
        Simple stopping rule based on power and significance.
        """
        
        status = "continue"
        recommendation = "Collect more samples."
        
        if current_n < min_n:
            status = "continue"
            recommendation = f"Current N ({current_n}) is below minimum required N ({min_n})."
        elif current_p_value < alpha and current_power >= target_power:
            status = "stop"
            recommendation = "Significant effect found with sufficient power. Experiment can be stopped."
        elif current_p_value > alpha and current_power >= target_power:
            status = "stop"
            recommendation = "No significant effect found, but power is sufficient. Further sampling is unlikely to change outcome."
        elif current_p_value > alpha and current_power < target_power:
            status = "continue"
            recommendation = f"No significant effect, but power ({current_power:.2f}) is below target ({target_power}). Collect more samples."
            
        return {
            "status": status,
            "recommendation": recommendation,
            "current_metrics": {
                "n": current_n,
                "p_value": current_p_value,
                "power": current_power
            }
        }
