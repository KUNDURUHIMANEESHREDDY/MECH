import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple

class DistributionDiagnostics:
    """
    Automatic detection of normality to guide statistical test selection.
    """

    @staticmethod
    def test_normality(data: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Tests for normality using Shapiro-Wilk and D'Agostino's K-squared tests.
        """
        n = len(data)
        
        # Shapiro-Wilk test (best for N < 5000)
        if n >= 3 and n <= 5000:
            stat_sw, p_sw = stats.shapiro(data)
            is_normal_sw = p_sw > alpha
        else:
            stat_sw, p_sw, is_normal_sw = None, None, None
            
        # D'Agostino's K-squared test (needs N >= 8)
        if n >= 8:
            stat_k2, p_k2 = stats.normaltest(data)
            is_normal_k2 = p_k2 > alpha
        else:
             stat_k2, p_k2, is_normal_k2 = None, None, None

        # Consensus (if either says not normal, treat as not normal to be safe, or vice versa)
        # We will use Shapiro if available, else K2
        is_normal = False
        if is_normal_sw is not None:
            is_normal = is_normal_sw
        elif is_normal_k2 is not None:
            is_normal = is_normal_k2

        return {
            "is_normal": is_normal,
            "shapiro_wilk": {"statistic": stat_sw, "p_value": p_sw, "is_normal": is_normal_sw},
            "dagostino_k2": {"statistic": stat_k2, "p_value": p_k2, "is_normal": is_normal_k2},
            "recommended_test": "t-test" if is_normal else "permutation"
        }

    @staticmethod
    def detect_outliers_iqr(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        mask = (data >= lower_bound) & (data <= upper_bound)
        return mask, ~mask
        
    @staticmethod
    def detect_outliers_mad(data: np.ndarray, threshold: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        # Modified Z-score
        if mad == 0:
             return np.ones(len(data), dtype=bool), np.zeros(len(data), dtype=bool)
             
        modified_z_scores = 0.6745 * (data - median) / mad
        mask = np.abs(modified_z_scores) <= threshold
        return mask, ~mask
        
    @classmethod
    def analyze_outliers(cls, data: np.ndarray, method: str = 'mad') -> Dict[str, Any]:
        if method == 'mad':
            keep, remove = cls.detect_outliers_mad(data)
        else:
            keep, remove = cls.detect_outliers_iqr(data)
            
        return {
            "method": method,
            "retained_count": int(np.sum(keep)),
            "removed_count": int(np.sum(remove)),
            "removed_indices": np.where(remove)[0].tolist()
        }
