from typing import Tuple
import numpy as np
from bonXAI.core.utils import median_pairwise_distance_sample


def resolve_kernel_params(name: str, X: np.ndarray, seed: int) -> Tuple[bytes, np.ndarray]:
    """
    Map kernel name to (kernel_type, k_params)
    """
    ell = median_pairwise_distance_sample(X, n_pairs=100_000, random_state=seed)

    if name == "gaussian":
        lam_sqd = ell**2
        print(f"Calculating Gaussian kernel with lambda^2 = {lam_sqd} ")
        return b"gaussian", np.array([lam_sqd])
    
    elif name == "sobolev":
        print("Calculating Sobolev kernel for smoothness: 1.0, 2.0, 3.0 - as a sum")
        k_params = np.array([1.0, 2.0, 3.0])
        return b"sobolev", k_params
    
    elif name == "inverse_multiquadric":
        print(f"Calculating Inverse-Multiquadric kernel for c = {ell**2} ")
        return b"inverse_multiquadric", np.array([ell**2])
    
    elif name == "matern":
        print("Calculating Matérn (sum of 3 kernels) ")
        k_params = np.array([1.0, ell * 0.5, 0.5, 1.0, ell * 1, 1.5, 1.0, ell * 2, 2.5], dtype=np.float64)
        return b"matern", k_params

    else:
        raise ValueError(f"Unknown kernel: {name}")
