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
from bonXAI.core.pytorch_ann import PyTorchANN
import shapiq
from torch.utils.data import DataLoader, TensorDataset
from pydvl.influence.torch import CgInfluence

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

    def __init__(self, model, explainer_name: str, strategy: str, task_type: str, seed: int = 0):
        self.model = model
        self.explainer_name = explainer_name.lower()
        self.strategy = strategy.lower()
        self.task_type = task_type.lower()
        self.seed = seed
# we need to add "na" option to strategy for shapiq, expected_gradients and influence and rewrite shao explainer to
# handle not if else but elifs + raise error if strategy not compatible with explainer
        if self.explainer_name not in {"shap", "sage", "shapiq", "expected_gradients", "influence"}:
            raise ValueError(f"Unsupported explainer: {self.explainer_name}")
        if self.strategy not in {"kernel", "permutation", "shapiq", "expected_gradients", "influence"}:
            raise ValueError(f"Unsupported strategy: {self.strategy}")
        if self.explainer_name == "expected_gradients" and self.strategy != "expected_gradients":
            raise ValueError("For expected_gradients explainer, strategy must be 'expected_gradients'.")
        
        if task_type == "classification" and hasattr(model, "predict_proba"):
            self.prediction_function = model.predict_proba
            self.loss = "cross entropy"
        elif task_type == "regression" and hasattr(model, "predict"):
            self.prediction_function = model.predict 
            self.loss = "mse"  
        else:
            raise ValueError("Model must have 'predict_proba' for classification or 'predict' for regression.")

    def explain(
        self,
        X_foreground: np.ndarray,
        X_background: np.ndarray,
        n_jobs: int,
        y_foreground: Optional[np.ndarray] = None,
        y_background: Optional[np.ndarray] = None
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
        elif self.explainer_name == "expected_gradients":
            return self._explain_expected_gradients(X_background=X_background, X_foreground=X_foreground, n_jobs=n_jobs)
        elif self.explainer_name == "shapiq":
            return self._explain_shapiq(X_background=X_background, X_foreground=X_foreground, n_jobs=n_jobs)
        elif self.explainer_name == "influence":
            if y_foreground is None or y_background is None:
                raise ValueError("Influence explanation requires both foreground and background labels (y).")
            return self._explain_influence(X_background=X_background, y_background=y_background, X_foreground=X_foreground, y_foreground=y_foreground)
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

        if self.task_type == "classification":
            predictions = self.prediction_function(X_foreground)
            predicted_classes = np.argmax(predictions, axis=1)
            num_samples = X_foreground.shape[0]
            shap_values = shap_values[np.arange(num_samples), :, predicted_classes]

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
    
    def _explain_shapiq(
            self,
            X_background: np.ndarray,
            X_foreground: np.ndarray,
            n_jobs: int = None
        ) -> Tuple[np.ndarray, float]:
        print(f"Explaining with ShapIQ. {len(X_foreground)} samples to explain using {len(X_background)} background samples.")
        start = time.time()
        if self.task_type == "regression":
            model_func = self.prediction_function
        elif self.task_type == "classification":
            def model_func(X):
                proba = self.prediction_function(X)
                preds = np.argmax(proba, axis=1)
                return np.array([proba[i, preds[i]] for i in range(len(preds))])

        imputer = shapiq.MarginalImputer(
            model=model_func,
            data=X_background,
            sample_size=len(X_background)
        )

        explainer = shapiq.TabularExplainer(
            model=model_func,
            data=X_background,
            approximator="regression",
            index="k-SII",
            max_order=2,
            imputer=imputer
        )
        main_effects = []
        pairwise_list = []

        for i in range(X_foreground.shape[0]):
            iv = explainer.explain(X_foreground[i], budget=1024, random_state=self.seed)
            main = np.asarray(iv.get_n_order_values(1)).ravel()
            main_effects.append(main)
            try:
                pair = iv.get_n_order_values(2)
            except Exception:
                pair = None
                print("No pairwise interactions found for sample:", i)
            pairwise_list.append(pair)

        self.main_effects = np.vstack(main_effects)
        elapsed = time.time() - start
        return pairwise_list, elapsed


    def _explain_expected_gradients(self,
            X_background: np.ndarray,
            X_foreground: np.ndarray,
            n_jobs: int = 8
        ):
        print(f"Explaining with Expected Gradients. {len(X_foreground)} samples to explain using {len(X_background)} background samples.")
        start = time.time()

        if not isinstance(self.model, PyTorchANN):
            raise TypeError("model must be an instance of PyTorchANN")

        inputs = torch.as_tensor(X_foreground, dtype=torch.float32)
        baselines = torch.as_tensor(X_background, dtype=torch.float32)
        explainer = captum.attr.IntegratedGradients(self.model.model_)

        explanations = []

        if self.task_type == "classification":
            predictions = self.prediction_function(X_foreground)
            predicted_classes = np.argmax(predictions, axis=1)

            for i, target_class in enumerate(predicted_classes):
                x_sample = inputs[i : i + 1]
                tasks = [
                    joblib.delayed(explainer.attribute)(
                        x_sample,
                        baselines[[j]],
                        target=int(target_class)
                    )
                    for j in range(baselines.shape[0])
                ]
                results = joblib.Parallel(n_jobs=n_jobs)(tasks)
                explanation = torch.mean(torch.stack(results), dim=0)
                explanations.append(explanation.detach().cpu().numpy().ravel())

        elif self.task_type == "regression":
            for i in range(inputs.shape[0]):
                x_sample = inputs[i : i + 1]
                tasks = [
                    joblib.delayed(explainer.attribute)(
                        x_sample,
                        baselines[[j]]
                    )
                    for j in range(baselines.shape[0])
                ]
                results = joblib.Parallel(n_jobs=n_jobs)(tasks)
                explanation = torch.mean(torch.stack(results), dim=0)
                explanations.append(explanation.detach().cpu().numpy().ravel())
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")

        final_explanations = torch.tensor(explanations, dtype=torch.float32)
        elapsed = time.time() - start

        return final_explanations, elapsed

    def _explain_influence(
            self,
            X_background: np.ndarray,
            y_background: np.ndarray,
            X_foreground: np.ndarray,
            y_foreground: np.ndarray
        ) -> Tuple[np.ndarray, float]:
        if isinstance(self.model, PyTorchANN):
            model = self.model.model_  
        elif isinstance(self.model, torch.nn.Module):
            model = self.model
        else:
            raise TypeError("Influence explainer requires a PyTorch model or PyTorchANN wrapper.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        start = time.time()

        X_train = torch.tensor(X_background).float().to(device)
        y_train = torch.tensor(y_background).long().to(device)

        X_test = torch.tensor(X_foreground).float().to(device)
        y_test = torch.tensor(y_foreground).long().to(device)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=False)

        if self.task_type == "classification":
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            loss_fn = torch.nn.MSELoss()

        influence_model = CgInfluence(
            model,
            loss_fn,
            regularization=1e-3,
            rtol=1e-7,
            atol=1e-7,
            solve_simultaneously=True
        ).fit(train_loader)

        influence_matrix = influence_model.influences(
            X_test, y_test, X_train, y_train, mode="up"
        ).cpu().numpy()

        elapsed = time.time() - start
        return influence_matrix, elapsed

