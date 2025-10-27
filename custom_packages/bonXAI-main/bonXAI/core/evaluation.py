import numpy as np
from typing import Dict
from bonXAI.core.metrics import compute_mae, compute_mmd, top_k_score

class Evaluator:
    def __init__(self, ground_truth_explanation: np.ndarray, ground_truth_points: np.ndarray):
        self.ground_truth_explanation = np.asarray(ground_truth_explanation)
        self.ground_truth_points = np.asarray(ground_truth_points)

    @staticmethod
    def _ensure_finite(arr: np.ndarray, name: str):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN/Inf")

    def evaluate_explanation(
        self,
        explanation: np.ndarray,
        time_elapsed: float,
        num_samples: int
    ) -> Dict[str, float]:
        explanation = np.asarray(explanation)
        if explanation.shape != self.ground_truth_explanation.shape:
            raise ValueError(
                f"explanation shape {explanation.shape} "
                f"!= ground truth {self.ground_truth_explanation.shape}"
            )
        if time_elapsed < 0:
            raise ValueError("time_elapsed must be non-negative")
        if not isinstance(num_samples, (int, np.integer)) or num_samples < 0:
            raise ValueError("num_samples must be a non-negative integer")
        self._ensure_finite(explanation, "explanation")
        self._ensure_finite(self.ground_truth_explanation, "ground truth explanation")

        mae = compute_mae(explanation, self.ground_truth_explanation)
        k = max(1, min(5, explanation.shape[-1])) 
        top_k = top_k_score(explanation, self.ground_truth_explanation, k=k)

        return {
            "mae": float(mae),
            "top_k": float(top_k),
            "explanation_time": float(time_elapsed),
            "size": int(num_samples),
        }

    def evaluate_compression(self, compressed_points: np.ndarray) -> Dict[str, float]:
        compressed_points = np.asarray(compressed_points)
        if compressed_points.ndim != 2 or self.ground_truth_points.ndim != 2:
            raise ValueError("points must be 2D arrays")
        if compressed_points.shape[1] != self.ground_truth_points.shape[1]:
            raise ValueError(
                f"Dim mismatch: compressed has {compressed_points.shape[1]} features, "
                f"ground truth has {self.ground_truth_points.shape[1]}"
            )
        self._ensure_finite(compressed_points, "compressed points")
        self._ensure_finite(self.ground_truth_points, "ground truth points")

        mmd = compute_mmd(self.ground_truth_points, compressed_points)
        return {"mmd": float(mmd)}
