import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score


def compute_mae(reference: np.ndarray, estimated: np.ndarray) -> float:
    """
    Computes Mean Absolute Error between two attribution arrays.
    """
    return mean_absolute_error(reference, estimated)


def compute_fidelity(original_preds: np.ndarray, explained_preds: np.ndarray) -> float:
    """
    Measures prediction agreement between original and explained outputs.
    """
    return accuracy_score(original_preds, explained_preds)


def compute_relative_error(reference: np.ndarray, estimated: np.ndarray) -> float:
    """
    Relative L1 error: ||ref - est||₁ / ||ref||₁
    """
    numerator = np.sum(np.abs(reference - estimated))
    denominator = np.sum(np.abs(reference)) + 1e-12  # avoid divide by zero
    return numerator / denominator
