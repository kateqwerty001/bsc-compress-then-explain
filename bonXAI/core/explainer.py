import time
import numpy as np
from typing import Optional, Tuple
import shap
import warnings
import sage

warnings.filterwarnings("ignore", category=UserWarning)


class Explainer:
    """
    Unified interface for SHAP and SAGE explanation methods.

    Parameters:
        model: classifier with predict_proba
        name: 'shap' or 'sage'
        variant: 'kernel' or 'permutation'
        seed: random seed
        n_jobs: parallelism (used only in SAGE permutation)
    """

    def __init__(self, model, name: str, variant: str = "kernel", seed: int = 0, n_jobs: int = 16):
        self.model = model
        self.name = name.lower()
        self.variant = variant.lower()
        self.seed = seed
        self.n_jobs = n_jobs

        if self.name not in {"shap", "sage"}:
            raise ValueError(f"Unsupported explainer: {self.name}")
        if self.variant not in {"kernel", "permutation"}:
            raise ValueError(f"Unsupported variant: {self.variant}")

    def explain(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Generate explanations.

        Parameters:
            X: compressed or selected data points
            y: ground truth labels (required for SAGE)

        Returns:
            (explanation_values, time_elapsed)
        """
        if self.name == "shap":
            return self._explain_shap(X)
        elif self.name == "sage":
            if y is None:
                raise ValueError("SAGE explanation requires labels (y).")
            return self._explain_sage(X, y)
        raise RuntimeError("Invalid configuration.")

    def _explain_shap(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        start = time.time()
        if self.variant == "kernel":
            explainer = shap.KernelExplainer(lambda x: self.model.predict_proba(x)[:, 1], X, seed=self.seed)
            shap_values = explainer(X, silent=True).values
        else:
            explainer = shap.Explainer(self.model.predict_proba, X, algorithm="permutation")
            shap_values = explainer(X, silent=True).values
        end = time.time()
        return shap_values, end - start

    def _explain_sage(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        start = time.time()
        imputer = sage.MarginalImputer(self.model.predict_proba, X)

        if self.variant == "permutation":
            estimator = sage.PermutationEstimator(
                imputer,
                loss="cross entropy",
                random_state=self.seed,
                n_jobs=self.n_jobs
            )
        else:
            estimator = sage.KernelEstimator(
                imputer,
                loss="cross entropy",
                random_state=self.seed
            )

        values = estimator(X, y, bar=False, verbose=False).values
        end = time.time()
        return values, end - start
