from typing import Tuple
import numpy as np


def resolve_kernel_params(name: str, X: np.ndarray) -> Tuple[bytes, np.ndarray]:
    """
    Map kernel name to (kernel_type, k_params)
    """
    if name == "gaussian":
        sigma = np.sqrt(2 * X.shape[1])
        return b"gaussian", np.array([sigma**2])
    elif name == "sobolev":
        return b"sobolev", np.array([1.0])
    elif name == "inverse_multiquadric":
        return b"inverse_multiquadric", np.array([1.0])
    else:
        raise ValueError(f"Unknown kernel: {name}")
