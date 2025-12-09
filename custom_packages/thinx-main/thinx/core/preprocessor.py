import math
import numpy as np
from typing import Optional, Tuple, Union, List
import time
from thinx.core.thinx_compress import thinx_compress
from thinx.core.kernel import resolve_kernel_params
from thinx.core.pytorch_nn import PyTorchNN
import torch
from torch.utils.data import TensorDataset, DataLoader
from pydvl.influence.torch import CgInfluence
from arfpy import arf as arf_mod
import pandas as pd
from stein_thinning.thinning import thin
from sklearn.mixture import GaussianMixture
from goodpoints import compress


class Compressor:
    """
    Compresses a dataset (X, y) using various methods:
    - Kernel Thinning
    - Stein Thinning
    - Compression based on Influence Functions
    - Compression based on ARFPy (ARF - Adversarial Random Forests)
    - IID sampling
    """
    def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            model,
            seed: int = 0
    ):
        """
        Initializes compressor with data and model.

        Args:
            X (np.ndarray): Array of samples.
            y (np.ndarray): Array of target values.
            model: Trained model.
            seed (int): Random seed for reproducibility.
        """
        self.X = X
        self.y = y
        self.model = model
        self.seed = seed

    def _kernel_thinning(
            self,
            g: int,
            num_bins: int,
            target_size: int,
            delta: float,
            kernel_type: bytes,
            k_params: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Compress data using kernel thinning.

        Args:
            g (int): Oversampling parameter for `compress_kt()`.
            num_bins (int): Number of bins for `compress_kt()`.
            target_size (int): Desired compressed coreset size.
            delta (float): Failure probability parameter for thinning.
            kernel_type (bytes): Kernel type (b"gaussian", b"sobolev", etc.).
            k_params (np.ndarray): Kernel parameters.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
            - Array of samples selected from X.
            - Array of target values corresponding to the selected samples.
            - Row indices (indices) in the original X used in the sample.
            - Compression time.
        """
        start = time.time()
        indices = thinx_compress(
            self.X, kernel_type, k_params, g=g, num_bins=num_bins, target_size=target_size, delta=delta, seed=self.seed
        )
        end = time.time()
        return self.X[indices], self.y[indices], indices, end - start

    def _stein_thinning(
            self,
            m: int,
            grad_type: bytes = b'gaussian',
            n_components: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Compress data using Stein thinning.

        Args:
            m (int): Number of points to select (target coreset size).
            grad_type (bytes): Type of gradient to use:
                - b'gaussian': Gaussian-based gradient
                - b'kde': Kernel density estimate gradient - experimental
                - b'gmm': Gaussian Mixture Model gradient - experimental
            n_components (int): Number of components for GMM (used if grad_type=b'gmm').

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
            - Array of samples selected from X.
            - Array of target values corresponding to the selected samples.
            - Row indices (indices) in the original X used in the sample.
            - Compression time.
        """
        grad = None
        start = time.time()

        if grad_type == b'gaussian':
            mu = self.X.mean(axis=0)
            Sigma = np.cov(self.X, rowvar=False) + 1e-6 * np.eye(self.X.shape[1])
            prec = np.linalg.inv(Sigma)
            grad = -(self.X - mu) @ prec

        elif grad_type == b'kde':
            n, d = self.X.shape
            std_dev = self.X.std(axis=0, ddof=1)
            bandwidth = np.mean(std_dev) * (4 / (d + 2) / n) ** (1 / (d + 4))  # a rule-of-thumb (Silverman-like)
            diffs = self.X[:, None, :] - self.X[None, :, :]
            sq_dist = np.sum(diffs ** 2, axis=2)
            weights = np.exp(-0.5 * sq_dist / bandwidth ** 2)
            grad = -np.einsum('ijk,ij->ik', diffs, weights) / (bandwidth ** 2 * n)

        elif grad_type == b'gmm':
            n, d = self.X.shape
            gmm = GaussianMixture(n_components=n_components, covariance_type='full')
            gmm.fit(self.X)

            mu = gmm.means_
            cov = gmm.covariances_
            prec = np.linalg.inv(cov)
            resp = gmm.predict_proba(self.X)

            grad = np.zeros((n, d))
            for k in range(n_components):
                diff = mu[k] - self.X
                grad += resp[:, [k]] * (diff @ prec[k])

        if grad is None:
            raise ValueError("Gradient must be provided for non-Gaussian Stein thinning.")

        indices = thin(self.X, grad, m)  # Stein thinning
        end = time.time()
        return self.X[indices], self.y[indices], indices, end - start



    def _influence_compression(
            self,
            target_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, object]:
        """
        Influence-based compression using PyDVL's CgInfluence.
        Computes self-influence scores for all samples in (X, y) and selects
        the top target_size most influential points as the compressed set.

        Args:
            target_size (int): Number of points to select for the compressed set.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
            - Array of samples selected from X.
            - Array of target values corresponding to the selected samples.
            - Row indices (indices) in the original X used in the sample.
            - Compression time.
            - Influence matrix (object) computed by CgInfluence
        """
        assert isinstance(self.model, torch.nn.Module), "Expected PyTorch model"
        n = target_size

        # --- Basic compression to sqrt(n'): n' - the largest power of 4 <= number os samples ---
        if n is None:
            x = int(np.floor(np.log2(np.sqrt(self.X.shape[0]))))
            n = 2 ** x

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self.model.to(device)
        model.eval()

        start = time.time()

        X_tensor = torch.tensor(self.X).float().to(device)
        y_tensor = torch.tensor(self.y).long().to(device)
        train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=128, shuffle=False)

        loss_fn = torch.nn.CrossEntropyLoss()

        influence_model = CgInfluence(
            model,
            loss_fn,
            regularization=1e-3,
            rtol=1e-7,
            atol=1e-7,
            solve_simultaneously=True
        ).fit(train_loader)

        # --- Self-influence: influence of each point on all others ---
        influence_matrix = influence_model.influences(X_tensor, y_tensor, X_tensor, y_tensor, mode="up")
        influence_avg = influence_matrix.abs().mean(axis=0).cpu().numpy()  # one score per training point

        # --- Select top-K by average influence magnitude ---
        top_k_indices = influence_avg.argsort()[-n:]
        end = time.time()
        return self.X[top_k_indices], self.y[top_k_indices], top_k_indices, end - start, influence_matrix

    def _arfpy_compression(
            self,
            target_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Generative compression using ARFPy (Adversarial Random Forests).
        Trains an ARF density model on (X, y) and synthesizes `target_size`
        new samples approximating the original data distribution.

        Args:
            target_size (int): Number of points to select for the compressed set.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
            - Array of samples selected from X.
            - Array of target values corresponding to the selected samples.
            - Row indices (indices) in the original X used in the sample.
            - Compression time.
        """
        n = target_size

        # --- Basic compression to sqrt(n'): n' - the largest power of 4 <= number os samples ---
        if n is None:
            x = int(np.floor(np.log2(np.sqrt(self.X.shape[0]))))
            n = 2 ** x

        start = time.time()
        df = pd.DataFrame(self.X, columns=[f"feat_{i}" for i in range(self.X.shape[1])])
        df["label"] = pd.Categorical(self.y)

        model = arf_mod.arf(x=df)
        model.forde()
        df_gen = model.forge(n=n)

        X_gen = df_gen.drop(columns=["label"]).to_numpy()
        y_gen = df_gen["label"].astype(int).to_numpy()

        end = time.time()

        indices = np.full(shape=(n,), fill_value=-1, dtype=int)
        return X_gen, y_gen, indices, end - start

    def _iid_sampling(
            self,
            target_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Compress data via IID random sampling without replacement.

        Args:
            target_size (int): Number of points to select for the compressed set.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
            - Array of samples selected from X.
            - Array of target values corresponding to the selected samples.
            - Row indices (indices) in the original X used in the sample.
            - Compression time.
        """
        n = target_size

        if n is None:
            n = self.X.shape[0]

        start = time.time()
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(self.X.shape[0], size=n, replace=False)
        end = time.time()
        return self.X[indices], self.y[indices], indices, end - start


class Preprocessor:
    """
        Preprocesses data using the selected modification and compression method.
    """
    def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            model,
            compression_method: str,
            data_modification_method: str = "none",
            seed: int = 0,
    ):
        """
        Initializes the Preprocessor.

        Args:
            X (np.ndarray): Array of samples to preprocess.
            y (np.ndarray): Array of target values.
            model (object): Trained model.
            compression_method (str): Name of the compression method to apply
                Options: {"kernel_thinning", "stein_thinning", "influence", "arfpy", "iid"}
            data_modification_method (str): Modifying data method to apply before compression.
                Options: {"none", "predictions", "stratified"}.
            seed (int): Random seed.
        """
        self.X = X
        self.y = y
        self.model = model
        self.data_modification_method = data_modification_method
        self.compression_method = compression_method
        self.seed = seed

    def _data_with_predictions(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Augments X with model predictions/probabilities if available.

        Returns:
            X_aug: Feature matrix with predictions/probabilities appended
            preds: Original predictions/probabilities
        """
        try:
            preds = self.model.predict_proba(self.X)
        except AttributeError:
            try:
                preds = self.model.predict(self.X)
            except AttributeError:
                raise AttributeError("Model has neither predict_proba nor predict method.")

        preds = np.array(preds).reshape(len(preds), -1)
        X_aug = np.concatenate([self.X, preds], axis=1)

        return X_aug, preds

    def _dispatch_compression(
            self,
            X_mod: np.ndarray,
            y_mod: np.ndarray,
            g: int,
            num_bins: int,
            target_size: int,
            kernel: str,
            delta: float,
            grad_type: bytes,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, float],
    Tuple[np.ndarray, np.ndarray, np.ndarray, float, object]]:
        """
        Dispatches the selected compression method and executes it on the modified dataset.

        Args:
            X_mod (np.ndarray): Array of samples after applying `data_modification_method`.
            y_mod (np.ndarray): Array of target values.
            g (int): Oversampling parameter in `thinx_compress()`.
            num_bins (int): Number of bins in `thinx_compress()`.
            target_size (int): Desired number of samples after compression.
            kernel (str): Kernel type used in kernel thinning.
                Options: {"gaussian", "sobolev", "ineverse_multiquadric", "matern"}
            delta (float): Delta parameter for kernel thinning.
            grad_type (bytes): Gradient type identifier for Stein thinning.
                Options: {"gaussian", "kde", "gmm"}. "kde" and "gmm" are experimental.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray, np.ndarray, float],
                  Tuple[np.ndarray, np.ndarray, np.ndarray, float, object]]:
            - Array of samples selected from X.
            - Array of target values corresponding to the selected samples.
            - Row indices (indices) in the original X used in the sample.
            - Compression time.
            - (Optional) Method-specific object (e.g., influence matrix)
        """
        compressor = Compressor(X_mod, y_mod, self.model, seed=self.seed)

        if self.compression_method == "kernel_thinning":
            kernel_type, k_params = resolve_kernel_params(kernel, X_mod, seed=self.seed)
            return compressor._kernel_thinning(g, num_bins, target_size, delta, kernel_type, k_params)

        elif self.compression_method == "stein_thinning":
            return compressor._stein_thinning(target_size, grad_type)

        elif self.compression_method == "influence":
            return compressor._influence_compression(target_size)

        elif self.compression_method == "arfpy":
            return compressor._arfpy_compression(target_size)

        elif self.compression_method == "iid":
            return compressor._iid_sampling(target_size)

        else:
            raise ValueError(f"Unknown compression method: {self.compression_method}")

    def preprocess(
            self,
            g: Optional[int] = 0,
            num_bins: Optional[int] = 4,
            target_size: Optional[int] = None,
            kernel: str = "gaussian",
            delta: float = 0.5,
            grad_type: bytes = b'gaussian',
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray, np.ndarray, float, object]]:
        """
        Runs the full preprocessing pipeline, including optional dataset
        modification and the selected compression method.

        Args:
            g (int): Oversampling parameter in `thinx_compress()`.
            num_bins (int): Number of bins in `thinx_compress()`.
            target_size (int): Desired number of samples after compression.
            kernel (str): Kernel type used in kernel thinning.
                Options: {"gaussian", "sobolev", "ineverse_multiquadric", "matern"}
            delta (float): Delta parameter for kernel thinning.
            grad_type (bytes): Gradient type identifier for Stein thinning.
                Options: {"gaussian", "kde", "gmm"}. "kde" and "gmm" are experimental.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray, np.ndarray, float],
                  Tuple[np.ndarray, np.ndarray, np.ndarray, float, object]]:
            - Array of samples selected from X.
            - Array of target values corresponding to the selected samples.
            - Row indices (indices) in the original X used in the sample.
            - Compression time.
            - (Optional) Method-specific object (e.g., influence matrix)

        Raises:
            ValueError: If an unknown data modification method is provided.
            ValueError: If the model does not implement `predict()` or `predict_proba()` in stratified mode.
            ValueError: If an unknown compression method is specified.
        """
        if self.data_modification_method in {"none", "stratified"}:
            X_mod, y_mod = self.X, self.y

        elif self.data_modification_method == "predictions":
            X_mod, _ = self._data_with_predictions()
            y_mod = self.y

        else:
            raise ValueError(f"Unknown data modification method: {self.data_modification_method}")

        if self.data_modification_method == "stratified":
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(self.X)
                y_pred_cls = np.argmax(proba, axis=1)
            elif hasattr(self.model, "predict"):
                y_pred_cls = self.model.predict(self.X)
            else:
                raise ValueError("model must implement `predict()` or `predict_proba()` for 'stratified' mode.")

            X_parts: List[np.ndarray] = []
            y_parts: List[np.ndarray] = []
            idx_parts: List[np.ndarray] = []
            total_time = 0.0

            classes = np.unique(y_pred_cls)
            for cls in classes:
                mask = (y_pred_cls == cls)
                X_cur_class = X_mod[mask]
                y_cur_class = y_mod[mask]

                if self.compression_method != "kernel_thinning":
                    # we can compress to any other size
                    share = max(1, int(round(target_size * (mask.sum() / len(self.X)))))
                    target_size_for_this_class = share

                elif self.compression_method == "kernel_thinning":
                    # we can compress only to powers of 2
                    default_compression_size = int(math.sqrt(compress.largest_power_of_four(len(self.X))))
                    compression_coeff = target_size / default_compression_size

                    default_compression_size_for_this_class = int(math.sqrt(compress.largest_power_of_four(len(X_cur_class))))
                    target_size_for_this_class = default_compression_size_for_this_class * compression_coeff

                elif self.compression_method == "None":
                    raise ValueError("No compression method specified.")

                else:
                    raise ValueError(f"Unknown compression method: {self.compression_method}")

                result = self._dispatch_compression(
                    X_cur_class,
                    y_cur_class,
                    g=g,
                    num_bins=num_bins,
                    target_size=target_size_for_this_class,
                    kernel=kernel,
                    delta=delta,
                    grad_type=grad_type
                )

                if len(result) == 5:
                    X_red_c, y_red_c, idx_c_local, t_c, _ = result
                else:
                    X_red_c, y_red_c, idx_c_local, t_c = result

                global_idx = np.where(mask)[0][idx_c_local]

                X_parts.append(X_red_c)
                y_parts.append(y_red_c)
                idx_parts.append(global_idx)
                total_time += t_c

            X_out = np.concatenate(X_parts, axis=0)
            y_out = np.concatenate(y_parts, axis=0)
            idx_out = np.concatenate(idx_parts, axis=0)
            return X_out, y_out, idx_out, total_time

        return self._dispatch_compression(
            X_mod,
            y_mod,
            g=g,
            num_bins=num_bins,
            target_size=target_size,
            kernel=kernel,
            delta=delta,
            grad_type=grad_type
        ) 