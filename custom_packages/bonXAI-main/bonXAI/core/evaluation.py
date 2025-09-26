import numpy as np
from typing import Dict

from bonXAI.core.metrics import compute_mae, compute_mmd, top_k_score

class Evaluator:
    def __init__(self, ground_truth_explanation: np.ndarray, reference_points: np.ndarray):
        self.ground_truth_explanation = ground_truth_explanation
        self.reference_points = reference_points

    def evaluate_explanation(
        self,
        explanation: np.ndarray,
        time_elapsed: float,
        num_samples: int
    ) -> Dict[str, float]:
        mae = compute_mae(self.ground_truth_explanation, explanation)
        top_k = top_k_score(explanation, self.ground_truth_explanation, k=5)

        return {
            "mae": float(mae),
            "top_k": float(top_k),
            "explanation_time": float(time_elapsed),
            "size": int(num_samples)
        }

    def evaluate_compression(self, compressed_points: np.ndarray) -> Dict[str, float]:
        mmd = compute_mmd(self.reference_points, compressed_points)
        return {
            "mmd": float(mmd)
        }

