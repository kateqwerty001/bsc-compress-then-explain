import time
import numpy as np
from typing import Optional, Tuple
import shap
import warnings
import sys
import sage
import torch
import torch.nn.functional as F
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
sys.stdout.reconfigure(line_buffering=True)


class Explainer:
    """
    Unified interface for SHAP and SAGE explanation methods.

    Parameters:
        model: classifier with predict_proba (classification) or predict (regression)
        exapliner_name: 'shap' or 'sage'
        strategy: 'kernel' or 'permutation'
        seed: random seed

    """

    def __init__(self, model, explainer_name: str, strategy: str , seed: int = 0):
        self.model = model
        self.explainer_name = explainer_name.lower()
        self.strategy = strategy.lower()
        self.seed = seed

        if self.explainer_name not in {"shap", "sage"}:
            raise ValueError(f"Unsupported explainer: {self.explainer_name}")
        if self.strategy not in {"kernel", "permutation"}:
            raise ValueError(f"Unsupported strategy: {self.strategy}")
        
        if hasattr(model, "predict_proba"):
            self.prediction_function = lambda x: model.predict_proba(x)[:, 1] # binary classification
            self.loss = "cross entropy"
        elif hasattr(model, "predict"):
            self.prediction_function = model.predict 
            self.loss = "mse"  
        else:
            raise ValueError("Model must have either 'predict_proba' or 'predict' method.")

    def _torch_predict_proba(self, X_numpy):
        X_tensor = torch.tensor(X_numpy).float()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def explain(
        self,
        x_foreground: np.ndarray,
        x_background: np.ndarray,
        n_jobs: int,
        bg_threshold: int = None,
        y_foreground: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Generate explanations.

        Parameters:
            X_foreground: data points to explain
            X_background: background data for reference
            bg_threshold: if x_background data is larger than this threshold, it will be randomly downsampled to this size
            n_jobs: number of parallel jobs (for SAGE - sage library implementation, for SHAP - custom implementation)
            y: ground truth labels (required for SAGE)

        Returns:
            (explanation_values, time_elapsed)
        """
        if bg_threshold is not None and x_background.shape[0] > bg_threshold:
            np.random.seed(seed=self.seed)
            id_gt = np.random.choice(x_background.shape[0], size=bg_threshold, replace=False)
            x_background = x_background[id_gt]

        if self.explainer_name == "shap":
            return self._explain_shap(x_background=x_background, x_foreground=x_foreground, n_jobs=n_jobs)
        elif self.explainer_name == "sage":
            if y_foreground is None:
                raise ValueError("SAGE explanation requires labels (y).")
            return self._explain_sage(x_background=x_background, x_foreground=x_foreground, y_foreground=y_foreground, n_jobs=n_jobs)
        raise RuntimeError("Invalid configuration.")

    def _explain_shap(
            self,
            x_background: np.ndarray, 
            x_foreground: np.ndarray, 
            n_jobs: int = None, 
        ) -> Tuple[np.ndarray, float]:
        print(f"Explaining with SHAP. {len(x_foreground)} samples to explain using {len(x_background)} background samples.")

        start = time.time()
        if self.strategy == "kernel":
            explainer = shap.KernelExplainer(self.prediction_function, x_background, seed=self.seed)
        else:
            masker = shap.maskers.Independent(x_background, max_samples=x_background.shape[0])
            explainer = shap.PermutationExplainer(self.prediction_function, masker, seed=self.seed)
        initialization_time = time.time() - start

        total_explanation_time = 0.0

        if n_jobs is None:
            start = time.time()
            shap_values = explainer(X_foreground, silent=True)
            total_explanation_time = time.time() - start + initialization_time
            return shap_values.values, total_explanation_time
        else:
            BATCH_SIZE = 10
            batches = [x_foreground[(i*BATCH_SIZE):(i+1)*BATCH_SIZE] for i in range(int(1+x_foreground.shape[0]/BATCH_SIZE))]

            def run_batch(batch):
                nonlocal total_explanation_time
                start = time.time()
                shap_values = explainer(batch, silent=True).values
                batch_time = time.time() - start
                print("Another batch done...")
                return shap_values, batch_time
            
            results = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(run_batch)(batch) for batch in batches if batch.shape[0] > 0
            )

            shap_values = np.concatenate([sv for sv, _ in results], axis=0)
            total_explanation_time = sum(batch_time for _, batch_time in results) + initialization_time
            return shap_values, total_explanation_time


    def _explain_sage(
            self,
            x_background: np.ndarray,
            x_foreground: np.ndarray,
            y_foreground: np.ndarray, 
            n_jobs: int = 16
        ) -> Tuple[np.ndarray, float]:
        print(f"Explaining with SAGE. {len(x_foreground)} samples to explain using {len(x_background)} background samples.")
        
        start = time.time()
        imputer = sage.MarginalImputer(self.prediction_function, x_background)

        if self.loss == "cross entropy":
            y_foreground = (y_foreground == 1).astype(int)
            if len(np.unique(y_foreground)) == 1: # edge case: all same class => error in cross entropy loss
                y_foreground[0] = 1 - y_foreground[0]

        if self.strategy == "kernel":
            explainer = sage.KernelEstimator(imputer, loss=self.loss, random_state=self.seed)
        else:
            if n_jobs is None:
                explainer = sage.PermutationEstimator(imputer, loss=self.loss, random_state=self.random_state)
            else:
                explainer = sage.PermutationEstimator(imputer, loss=self.loss, random_state=self.seed, n_jobs=n_jobs)

        sage_values = explainer(x_foreground, y_foreground, bar=False, verbose=False).values
        end = time.time()
        return sage_values, end - start
