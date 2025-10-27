from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

def compute_mae(values1: np.ndarray, values2: np.ndarray) -> float:
    if values1.shape != values2.shape:
        raise ValueError(f"Shape mismatch: {values1.shape} vs {values2.shape}")
    if not np.isfinite(values1).all() or not np.isfinite(values2).all():
        raise ValueError("inputs must be finite")
    
    return np.mean(np.abs(values2 - values1))


def compute_mmd(X: np.ndarray, Y: np.ndarray, kernel="rbf", gamma=None) -> float:
    """Simplified unbiased MMD^2 with RBF kernel"""

    if gamma is None:
        gamma = 1.0 / X.shape[1]
    
    if X.shape[1] != Y.shape[1]:
        raise ValueError("feature dimensions must match")
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError("inputs must be finite")

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

    if exp.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {exp.shape} vs {gt.shape}")
    if exp.ndim not in (1, 2, 3):
        raise ValueError(f"Unsupported ndim={exp.ndim}. Must be 1D, 2D, or 3D.")
    if not (np.isfinite(exp).all() and np.isfinite(gt).all()):
        raise ValueError("Inputs must contain only finite values (no NaN/Inf).")

    if exp.ndim == 1:
        d = exp.shape[0]
    elif exp.ndim == 2:
        d = exp.shape[1]
    else: 
        d = exp.shape[1]
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an integer, got {type(k).__name__}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > d:
        k = d

    # --- Case 1: 3D SHAP arrays (n_samples, n_features, n_classes) - for SHAP multiclass
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
    # --- Case 2: 2D arrays (n_samples, n_features) - for SHAP single output
    elif exp.ndim == 2:
        n_samples, n_features = exp.shape
        overlaps = np.empty(n_samples, dtype=float)
        for i in range(n_samples):
            e = exp[i]
            g = gt[i]
            idx1 = np.argpartition(np.abs(e), -k)[-k:]
            idx2 = np.argpartition(np.abs(g), -k)[-k:]
            overlaps[i] = np.intersect1d(idx1, idx2).size / k
        return float(overlaps.mean())
    # --- Case 3: 1D arrays (n_features) - for SAGE
    elif exp.ndim == 1:
        e_imp = np.abs(exp)
        g_imp = np.abs(gt)
        top_exp = np.argsort(e_imp)[-k:]
        top_gt = np.argsort(g_imp)[-k:]
        overlap = len(set(top_exp.tolist()).intersection(set(top_gt.tolist())))
        return overlap / k
    
def topk_pair_overlap(pairs_A, pairs_B, k: int = 5) -> float:
    if pairs_A.shape != pairs_B.shape or pairs_A.ndim != 3 or pairs_A.shape[1] != pairs_A.shape[2]:
        raise ValueError("A and B must be (n, d, d) with same shape")

    A = np.stack(pairs_A) if isinstance(pairs_A, (list, tuple)) else np.asarray(pairs_A)
    B = np.stack(pairs_B) if isinstance(pairs_B, (list, tuple)) else np.asarray(pairs_B)
    n, d, _ = A.shape
    if d < 2:
        raise ValueError("d must be >= 2 for pair overlap")
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        raise ValueError("inputs must be finite")
    iu = np.triu_indices(d, k=1)
    m = iu[0].size
    kk = min(max(1, k), m)
    overlaps = np.empty(n, dtype=float)
    for s in range(n):
        a = np.abs(A[s])[iu]
        b = np.abs(B[s])[iu]
        if np.allclose(a, b, atol=1e-12):
            overlaps[s] = 1.0
            continue
        ath = np.partition(a, -kk)[-kk]
        bth = np.partition(b, -kk)[-kk]

        topA = np.flatnonzero(a >= ath)
        topB = np.flatnonzero(b >= bth)

        denom = max(1, min(len(topA), len(topB)))
        overlaps[s] = len(np.intersect1d(topA, topB)) / denom
    return float(overlaps.mean())

