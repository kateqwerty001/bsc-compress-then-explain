import numpy as np
from goodpoints_ext.compress import compresspp_kt


def iid_sampling(X: np.ndarray, y: np.ndarray, n: int, seed: int = 0):
    """
    Uniformly samples `n` points from the dataset.

    Args:
        X: Feature matrix
        y: Labels
        n: Number of points to sample
        seed: Random seed

    Returns:
        (X_subset, y_subset)
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx], y[idx]


def cte(X: np.ndarray, y: np.ndarray, kernel_type: bytes, k_params: np.ndarray,
        g: int, num_bins: int, seed: int = 0):
    """
    Compress-then-explain strategy on raw features only.

    Returns:
        (X_compressed, y_compressed)
    """
    idx = compresspp_kt(X, kernel_type=kernel_type, k_params=k_params,
                        g=g, num_bins=num_bins, seed=seed)
    return X[idx], y[idx]


def cte_with_predictions(X: np.ndarray, y: np.ndarray, pred: np.ndarray,
                         kernel_type: bytes, k_params: np.ndarray,
                         g: int, num_bins: int, seed: int = 0):
    """
    Compress features concatenated with model predictions.

    Args:
        pred: Predictions as np.ndarray of shape (n_samples, 1) or logits

    Returns:
        (X_compressed, y_compressed)
    """
    X_aug = np.concatenate([X, pred], axis=1)
    idx = compresspp_kt(X_aug, kernel_type=kernel_type, k_params=k_params,
                        g=g, num_bins=num_bins, seed=seed)
    return X[idx], y[idx]


def cte_stratified(X: np.ndarray, y: np.ndarray, pred: np.ndarray,
                   kernel_type: bytes, k_params: np.ndarray,
                   g: int, num_bins: int, seed: int = 0):
    """
    Stratified CTE: compress points class-by-class based on predictions.

    Returns:
        (X_compressed, y_compressed)
    """
    unique_classes = np.unique(pred)
    final_indices = []

    for cls in unique_classes:
        cls_mask = pred == cls
        X_cls = X[cls_mask]
        y_cls = y[cls_mask]

        if X_cls.shape[0] < g:
            continue  # skip if not enough samples

        idx_local = compresspp_kt(X_cls, kernel_type=kernel_type, k_params=k_params,
                                  g=g, num_bins=num_bins, seed=seed)
        idx_global = np.where(cls_mask)[0][idx_local]
        final_indices.extend(idx_global)

    final_indices = np.array(final_indices)
    return X[final_indices], y[final_indices]
