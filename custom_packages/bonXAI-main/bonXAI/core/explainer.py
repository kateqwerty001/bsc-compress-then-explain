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
import captum
from shapiq.explainer.tabular import TabularExplainer as ShapiqTabularExplainer
from bonXAI.core.pytorch_ann import PyTorchANN

warnings.filterwarnings("ignore", category=UserWarning)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


class Explainer:
    """
    Unified interface for SHAP, SAGE and Shapiq explanation methods.

    Parameters:
        model: classifier with predict_proba (classification) or predict (regression)
        exapliner_name: 'shap' or 'sage' or 'shapiq' or 'expected_gradients'
        strategy: 'kernel' or 'permutation' or 'expected_gradients'
        seed: random seed

    """

    def __init__(self, model, explainer_name: str, strategy: str , seed: int = 0):
        self.model = model
        self.explainer_name = explainer_name.lower()
        self.strategy = strategy.lower()
        self.seed = seed

        if self.explainer_name not in {"shap", "sage", "shapiq", "expected_gradients"}:
            raise ValueError(f"Unsupported explainer: {self.explainer_name}")
        if self.strategy not in {"kernel", "permutation", "expected_gradients"}:
            raise ValueError(f"Unsupported strategy: {self.strategy}")
        if self.explainer_name == "expected_gradients" and self.strategy != "expected_gradients":
            raise ValueError("For expected_gradients explainer, strategy must be 'expected_gradients'.")
        
        if hasattr(model, "predict_proba"):
            self.prediction_function = model.predict_proba
            self.loss = "cross entropy"
        elif hasattr(model, "predict"):
            self.prediction_function = model.predict 
            self.loss = "mse"  
        else:
            raise ValueError("Model must have either 'predict_proba' or 'predict' method.")
        
        self._shapiq_explainer = None
        self._shapiq_index = "k-SII"      
        self._shapiq_max_order = 2
        self._shapiq_imputer = "marginal" 
        self._shapiq_budget = 1024
        self._shapiq_class_index = 1  
        self.shapiq_pairwise_ = None


    def _torch_predict_proba(self, X_numpy):
        X_tensor = torch.tensor(X_numpy).float()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def explain(
        self,
        X_foreground: np.ndarray,
        X_background: np.ndarray,
        n_jobs: int,
        y_foreground: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Generate explanations.

        Parameters:
            X_foreground: data points to explain
            X_background: background data for reference
            n_jobs: number of parallel jobs (for SAGE - sage library implementation, for SHAP - custom implementation)
            y: ground truth labels (required for SAGE)

        Returns:
            (explanation_values, time_elapsed)
        """
        if self.explainer_name == "shap":
            return self._explain_shap(X_background=X_background, X_foreground=X_foreground, n_jobs=n_jobs)
        elif self.explainer_name == "sage":
            if y_foreground is None:
                raise ValueError("SAGE explanation requires labels (y).")
            return self._explain_sage(X_background=X_background, X_foreground=X_foreground, y_foreground=y_foreground, n_jobs=n_jobs)
        elif self.explainer_name == "shapiq":
            return self._explain_shapiq(X_background=X_background, X_foreground=X_foreground)
        elif self.explainer_name == "expected_gradients":
            return self._explain_expected_gradients(X_background=X_background, X_foreground=X_foreground, n_jobs=n_jobs)
        raise RuntimeError("Invalid configuration.")

    def _explain_shap(
            self,
            X_background: np.ndarray, 
            X_foreground: np.ndarray, 
            n_jobs: int = None, 
        ) -> Tuple[np.ndarray, float]:
        print(f"Explaining with SHAP. {len(X_foreground)} samples to explain using {len(X_background)} background samples.")

        start = time.time()
        if self.strategy == "kernel":
            explainer = shap.KernelExplainer(self.prediction_function, X_background, seed=self.seed)
        else:
            masker = shap.maskers.Independent(X_background, max_samples=X_background.shape[0])
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
            batches = [X_foreground[(i*BATCH_SIZE):(i+1)*BATCH_SIZE] for i in range(int(1+X_foreground.shape[0]/BATCH_SIZE))]

            def run_batch(batch):
                nonlocal total_explanation_time
                start = time.time()
                shap_values = explainer(batch, silent=True).values
                batch_time = time.time() - start
                return shap_values, batch_time
            
            results = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(run_batch)(batch) for batch in batches if batch.shape[0] > 0
            )

            shap_values = np.concatenate([sv for sv, _ in results], axis=0)
            total_explanation_time = sum(batch_time for _, batch_time in results) + initialization_time
            return shap_values, total_explanation_time


    def _explain_sage(
            self,
            X_background: np.ndarray,
            X_foreground: np.ndarray,
            y_foreground: np.ndarray, 
            n_jobs: int = 16
        ) -> Tuple[np.ndarray, float]:
        print(f"Explaining with SAGE. {len(X_foreground)} samples to explain using {len(X_background)} background samples.")
        
        start = time.time()
        imputer = sage.MarginalImputer(self.prediction_function, X_background)

        if self.loss == "cross entropy":
            y_foreground = (y_foreground == 1).astype(int)
            if len(np.unique(y_foreground)) == 1:
                y_foreground[0] = 1 - y_foreground[0]

        if self.strategy == "kernel":
            explainer = sage.KernelEstimator(imputer, loss=self.loss, random_state=self.seed)
        else:
            if n_jobs is None:
                explainer = sage.PermutationEstimator(imputer, loss=self.loss, random_state=self.seed)
            else:
                explainer = sage.PermutationEstimator(imputer, loss=self.loss, random_state=self.seed, n_jobs=n_jobs)

        sage_values = explainer(X_foreground, y_foreground, bar=False, verbose=False).values
        end = time.time()
        elapsed = end - start
        return sage_values, elapsed

    def _ensure_shapiq(self, background_X: np.ndarray):
        if self._shapiq_explainer is not None:
            return
        if hasattr(self.model, "predict_proba"):
            model_fn = lambda x: self.prediction_function(x)[:, self._shapiq_class_index]
        else:
            model_fn = lambda x: np.asarray(self.prediction_function(x)).ravel()

        self._shapiq_explainer = ShapiqTabularExplainer(
            model=model_fn,
            data=np.asarray(background_X),
            imputer=self._shapiq_imputer,
            index=self._shapiq_index,
            max_order=self._shapiq_max_order,
            random_state=self.seed,
            verbose=False,
        )

    def _explain_shapiq(
        self,
        X_background: np.ndarray,
        X_foreground: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        print(f"Explaining with SHAPIQ (k-SII, max_order=2). "
              f"{len(X_foreground)} samples to explain using {len(X_background)} background samples.")

        start = time.time()
        self._ensure_shapiq(X_background)

        main_effects = []
        pairwise_list = []

        for i in range(X_foreground.shape[0]):
            iv = self._shapiq_explainer.explain_function(
                x=X_foreground[i : i + 1],
                budget=self._shapiq_budget,
                random_state=self.seed,
            )
            main = np.asarray(iv.get_n_order_values(1)).ravel()
            try:
                pair = iv.get_n_order_values(2)
            except Exception:
                pair = None
            main_effects.append(main)
            pairwise_list.append(pair)

        elapsed = time.time() - start
        main_effects = np.vstack(main_effects)
        self.shapiq_pairwise_ = pairwise_list
        return main_effects, elapsed
    
    def _explain_expected_gradients(self,
            X_background: np.ndarray,
            X_foreground: np.ndarray,
            n_jobs: int = 16
        ):
        print(f"Explaining with Expected Gradients. {len(X_foreground)} samples to explain using {len(X_background)} background samples.")
        start = time.time()

        if not isinstance(self.model, PyTorchANN):
            raise TypeError("model must be an instance of PytorchANN")
        
        num_classes = self.model.model_.network[-1].out_features

        explainer = captum.attr.IntegratedGradients(self.model.model_)
        explanations_per_class = []
        for class_index in range(num_classes):
            tasks = [
                joblib.delayed(explainer.attribute)(X_foreground, X_background[[i]], target=class_index) 
                for i in range(X_background.shape[0])
            ]
            results = joblib.Parallel(n_jobs=n_jobs)(tasks)
            explanation_for_one_class = torch.mean(torch.stack(results), dim=0)

            explanations_per_class.append(explanation_for_one_class)

        final_explanations = torch.stack(explanations_per_class, dim=2)

        elapsed = time.time() - start

        return final_explanations, elapsed