# import numpy as np
# from typing import Optional, Tuple, Union
# import time
# from bonXAI.core.goodpoints_ext.compress import compresspp_kt, compress_kt
# from bonXAI.core.kernel import resolve_kernel_params
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader
# from pydvl.influence.torch import CgInfluence
# from arfpy import arf as arf_mod
# import pandas as pd


# class Preprocessor:
#     """
#     Handles different preprocessing strategies:
#     - IID sampling
#     - CTE (compress-then-explain)
#     - CTE with predictions
#     - Stratified compression
#     - Identity (no change)
#     """

#     def __init__(
#         self,
#         method: str,
#         model=None,
#         g: Optional[int] = None,
#         num_bins: Optional[int] = None,
#         target_size: Optional[int] = None,
#         kernel: str = "gaussian",
#         seed: int = 0
#     ):
#         self.method = method.lower()
#         self.model = model
#         self.g = g
#         self.num_bins = num_bins
#         self.target_size = target_size
#         self.kernel = kernel
#         self.seed = seed

#     def run(self, X: np.ndarray, y: np.ndarray) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray, np.ndarray, float, object]]:
#         """
#         Executes the selected preprocessing method.

#         Returns:
#             X_red, y_red, indices: compressed samples and their original indices
#         """
#         rng = np.random.default_rng(self.seed)
#         kernel_type, k_params = resolve_kernel_params(self.kernel, X)

#         if self.method == "none":
#             indices = np.arange(X.shape[0])
#             return X, y, indices, 0.0

#         elif self.method == "iid":
#             n = self.target_size
#             if n is None: # in case of compresspp_kt() comparison
#                 x = int(np.floor(np.log2(np.sqrt(X.shape[0]))))
#                 n = 2 ** x
#             start = time.time()
#             indices = rng.choice(X.shape[0], size=n, replace=False)
#             end = time.time()
#             return X[indices], y[indices], indices, end - start

#         elif self.method == "compress":
#             return self._compress_core(X, y, kernel_type, k_params)

#         elif self.method == "compress_with_predictions":
#             if self.model is None:
#                 raise ValueError("Model is required for compress_with_predictions.")
#             preds = self.model.predict_proba(X)
#             X_aug = np.concatenate([X, preds], axis=1)
#             _, y_result, indices_result, time_result = self._compress_core(X_aug, y, kernel_type, k_params)
#             return X[indices_result], y_result, indices_result, time_result
        
#         elif self.method == "compress_stratified":
#             if self.model is None:
#                 raise ValueError("Model is required for compress_stratified.")
#             pred_classes = np.argmax(self.model.predict_proba(X), axis=1)
#             return self._compress_stratified(X, y, pred_classes, kernel_type, k_params)

#         elif self.method == "influence":
#             return self._run_influence_compression(X, y)
        
#         elif self.method == "arfpy":
#             return self._run_arfpy_compression(X, y)

#         else:
#             raise ValueError(f"Unsupported preprocessing method: {self.method}")

#     def _compress_core(
#         self, X: np.ndarray, y: np.ndarray, kernel_type: bytes, k_params: np.ndarray
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
#         start = time.time()
#         if self.g is not None or self.num_bins is not None:
#             indices = compress_kt(X, kernel_type, k_params, g=self.g or 4, num_bins=self.num_bins or 4, seed=self.seed)
#         else:
#             indices = compresspp_kt(X, kernel_type, k_params, seed=self.seed)
#         end = time.time()
#         return X[indices], y[indices], indices, end - start

#     def _compress_stratified(
#         self, X: np.ndarray, y: np.ndarray, pred: np.ndarray,
#         kernel_type: bytes, k_params: np.ndarray
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
#         final_indices = []
#         start = time.time()
#         for cls in np.unique(pred):
#             mask = pred == cls
#             X_cls = X[mask]

#             if self.g is not None or self.num_bins is not None:
#                 idx = compress_kt(X_cls, kernel_type, k_params, g=self.g or 4, num_bins=self.num_bins or 4, seed=self.seed)
#             else:
#                 idx = compresspp_kt(X_cls, kernel_type, k_params, seed=self.seed)

#             global_idx = np.where(mask)[0][idx]
#             final_indices.extend(global_idx)
#         end = time.time()
#         final_indices = np.array(final_indices)
#         return X[final_indices], y[final_indices], final_indices, end - start
    
#     def _run_influence_compression(self, X, y):
#         """
#         Influence-based compression using PyDVL's CgInfluence.
#         Computes self-influence scores for all samples in (X, y) and selects
#         the top target_size most influential points as the compressed set.
#         """
#         assert isinstance(self.model, torch.nn.Module), "Expected PyTorch model"
#         n = self.target_size
#         if n is None:
#             x = int(np.floor(np.log2(np.sqrt(X.shape[0]))))
#             n = 2 ** x

#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model = self.model.to(device)
#         model.eval()

#         start = time.time()

#         X_tensor = torch.tensor(X).float().to(device)
#         y_tensor = torch.tensor(y).long().to(device)
#         train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=128, shuffle=False)

#         loss_fn = torch.nn.CrossEntropyLoss()

#         influence_model = CgInfluence(
#             model,
#             loss_fn,
#             regularization=1e-3,
#             rtol=1e-7,
#             atol=1e-7,
#             solve_simultaneously=True
#         ).fit(train_loader)

#         # Self-influence: influence of each point on all others
#         influence_matrix = influence_model.influences(X_tensor, y_tensor, X_tensor, y_tensor, mode="up")
#         influence_avg = influence_matrix.mean(axis=0).cpu().numpy()  # one score per training point

#         # Select top-K by average influence magnitude
#         top_k_indices = influence_avg.argsort()[-n:]
#         end = time.time()
#         return X[top_k_indices], y[top_k_indices], top_k_indices, end - start, influence_matrix
    
#     def _run_arfpy_compression(self, X: np.ndarray, y: np.ndarray):
#         """
#         Generative compression using ARFPy (Adversarial Random Forests).
#         Trains an ARF density model on (X, y) and synthesizes target_size new samples approximating the original data distribution.
#         """
#         start = time.time()

#         n = self.target_size
#         if n is None:
#             x = int(np.floor(np.log2(np.sqrt(X.shape[0]))))
#             n = 2 ** x
#         df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
#         df["label"] = pd.Categorical(y)

#         model = arf_mod.arf(x=df)
#         model.forde()                
#         df_gen = model.forge(n=n)    

#         X_gen = df_gen.drop(columns=["label"]).to_numpy()
#         y_gen = df_gen["label"].astype(int).to_numpy()

#         end = time.time()

#         indices = np.full(shape=(n,), fill_value=-1, dtype=int)
#         return X_gen, y_gen, indices, end - start
