from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

def compute_mae(values1: np.ndarray, values2: np.ndarray) -> float:
    if values1.shape != values2.shape:
        raise ValueError(f"Shape mismatch: {values1.shape} vs {values2.shape}")
    
    return np.mean(np.abs(values2 - values1))


def compute_mmd(X: np.ndarray, Y: np.ndarray, kernel="rbf", gamma=None) -> float:
    """Simplified unbiased MMD^2 with RBF kernel"""

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)

    return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)


def top_k_score(exp, gt, k=5):
    """
    Top-k agreement score between explanation and ground truth.
    Returns the proportion of overlapping features in the top-k.
    
    If explanations are multidimensional (e.g., [n_features, n_classes]),
    their absolute values are summed across axis=1 to get per-feature importance.
    """
    exp = np.asarray(exp)
    gt = np.asarray(gt)

    # --- Case 1: 3D SHAP arrays (n_samples, n_features, n_classes)
    if exp.ndim == 3:
        scores = []
        for e_sample, g_sample in zip(exp, gt):
            e_imp = np.sum(np.abs(e_sample), axis=-1)
            g_imp = np.sum(np.abs(g_sample), axis=-1)

            top_exp = np.argsort(e_imp)[-k:]
            top_gt = np.argsort(g_imp)[-k:]

            overlap = len(set(top_exp.tolist()).intersection(set(top_gt.tolist())))
            scores.append(overlap / k)
        return np.mean(scores)
    # --- Case 2: 2D arrays (n_samples, n_features)
    elif exp.ndim == 2:
        raise NotImplementedError("2D arrays  are not supported yet.")
    # --- Case 3: 1D arrays (n_features)
    elif exp.ndim == 1:
        e_imp = np.abs(exp)
        g_imp = np.abs(gt)
        top_exp = np.argsort(e_imp)[-k:]
        top_gt = np.argsort(g_imp)[-k:]
        overlap = len(set(top_exp.tolist()).intersection(set(top_gt.tolist())))
        return overlap / k

