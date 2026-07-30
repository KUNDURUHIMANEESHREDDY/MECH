import numpy as np
from typing import Dict, Any, List, Optional
from .diagnostics import DistributionDiagnostics
from .permutation_engine import PermutationEngine
from .hypothesis_testing import HypothesisTesting
from .power_analysis import PowerAnalysis
from .effect_sizes import EffectSizes
from .statistical_trace import StatisticalTrace
from .statistical_quality import StatisticalQualityScore
from .statistical_protocol import StatisticalProtocol
from .bayesian_frequentist import BayesianFrequentistComparison

class StatisticalValidator:
    """
    Orchestrates the statistical modules.
    Automatically chooses appropriate tests based on diagnostics.
    """
    def __init__(self, experiment_id: str, protocol: Optional[StatisticalProtocol] = None):
        self.trace = StatisticalTrace(experiment_id)
        self.protocol = protocol or StatisticalProtocol()
        self.trace.add_step("Protocol Loaded", self.protocol.to_dict())

    def compare_groups(self, group_a: np.ndarray, group_b: np.ndarray, feature_name: str = "feature") -> Dict[str, Any]:
        """
        End-to-end comparison between two groups with automatic test selection.
        """
        self.trace.add_step("Start Comparison", {"feature": feature_name, "n_a": len(group_a), "n_b": len(group_b)})
        
        # 1. Diagnostics (Normality)
        diag_a = DistributionDiagnostics.test_normality(group_a)
        diag_b = DistributionDiagnostics.test_normality(group_b)
        
        is_normal = diag_a["is_normal"] and diag_b["is_normal"]
        self.trace.add_step("Normality Check", {"group_a": diag_a, "group_b": diag_b, "both_normal": is_normal})
        
        # 2. Test Selection & Execution
        if is_normal:
            test_used = "t-test"
            from scipy import stats
            stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False) # Welch's
            self.trace.add_step("Test Execution", {"test": "Welch's t-test", "statistic": float(stat), "p_value": float(p_val)})
        else:
            test_used = "permutation"
            # Define mean difference as statistic
            def diff_means(a, b): return np.mean(a) - np.mean(b)
            engine = PermutationEngine(n_permutations=self.protocol.permutation_count, seed=self.protocol.random_seed)
            res = engine.run_permutation_test(group_a, group_b, diff_means, permutation_type="label")
            p_val = res["p_value"]
            self.trace.add_step("Test Execution", {"test": "Permutation", "n_permutations": self.protocol.permutation_count, "p_value": p_val})
            
        # 3. Effect Size
        effect_sizes = EffectSizes.compute_all(group_a, group_b)
        self.trace.add_step("Effect Sizes", effect_sizes)
        
        # 4. Power Analysis
        current_n = min(len(group_a), len(group_b)) # Conservative estimate
        power_res = PowerAnalysis.analyze(effect_sizes[self.protocol.effect_size_metric], current_n, target_power=self.protocol.power_threshold)
        self.trace.add_step("Power Analysis", power_res)
        
        # 5. Bayesian vs Frequentist Comparison
        bayes_freq = BayesianFrequentistComparison.analyze(group_a, group_b, self.protocol.alpha)
        self.trace.add_step("Bayesian Analysis", bayes_freq)
        
        results = {
            "feature": feature_name,
            "test_used": test_used,
            "p_value_raw": p_val,
            "effect_sizes": effect_sizes,
            "power": power_res,
            "diagnostics": {"group_a": diag_a, "group_b": diag_b},
            "bayesian_frequentist": bayes_freq
        }
        
        # Calculate Quality Score
        quality = StatisticalQualityScore.evaluate(results)
        results["quality_score"] = quality
        
        return results

    def process_multiple_comparisons(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies multiple hypothesis correction to a list of comparison results.
        """
        raw_p = [res["p_value_raw"] for res in results]
        correction = HypothesisTesting.apply_correction(raw_p, method=self.protocol.correction_method, alpha=self.protocol.alpha)
        
        self.trace.add_step("Multiple Correction", {"method": self.protocol.correction_method, "num_rejected": correction["num_rejected"]})
        
        for i, res in enumerate(results):
            res["p_value_adjusted"] = correction["adjusted_p_values"][i]
            res["is_significant"] = correction["rejected"][i]
            res["correction_method"] = self.protocol.correction_method
            
            # Recalculate quality score with correction info
            res["quality_score"] = StatisticalQualityScore.evaluate(res)
            
        return results
        
    def get_trace_report(self) -> Dict[str, Any]:
        return self.trace.generate_report()
