from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

def compute_mae(reference: np.ndarray, estimate: np.ndarray) -> float:
    return mean_absolute_error(reference, estimate)

def compute_mmd(X: np.ndarray, Y: np.ndarray, kernel="rbf", gamma=None) -> float:
    """Simplified unbiased MMD^2 with RBF kernel"""

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)

    m = X.shape[0]
    n = Y.shape[0]
    return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)

def top_k_score(exp, gt, k=5):
    """
    Top-k agreement score between explanation and ground truth.
    Returns the proportion of overlapping features in the top-k.
    """
    top_exp = np.argsort(np.abs(exp))[-k:]
    top_gt = np.argsort(np.abs(gt))[-k:]
    overlap = len(set(top_exp).intersection(set(top_gt)))
    return overlap / k