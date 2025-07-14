import numpy as np
from typing import Tuple, Optional


def resolve_kernel_params(
    kernel_name: str,
    X: Optional[np.ndarray] = None,
    param: Optional[float] = None
) -> Tuple[bytes, np.ndarray]:
    """
    Map a human‑readable kernel name to the GoodPoints byte identifier
    and its parameter vector.

    Parameters
    ----------
    kernel_name : str
        'gaussian', 'sobolev', or 'inverse_multiquadric'.
    X : np.ndarray, optional
        Feature matrix – required for Gaussian so σ² can scale with dimension.
    param : float, optional
        Manually supplied kernel parameter.  If given, overrides any default.

    Returns
    -------
    kernel_type : bytes
        Byte tag used by compresspp_kt.
    k_params : np.ndarray
        One‑dimensional parameter vector expected by compresspp_kt.
    """
    k = kernel_name.lower()

    if k == "gaussian":
        if param is None:
            if X is None:
                raise ValueError(
                    "Feature matrix X must be supplied for Gaussian kernel "
                    "unless param is given explicitly."
                )
            # σ² = 2 · d  (d = number of features)
            param = 2.0 * X.shape[1]
        kernel_type = b"gaussian"

    elif k == "sobolev":
        # Sobolev smoothness exponent (s). Default = 1.0
        param = 1.0 if param is None else param
        kernel_type = b"sobolev"

    elif k in ("inverse_multiquadric", "imq"):
        # β parameter of the IMQ kernel. Default = 1.0
        param = 1.0 if param is None else param
        kernel_type = b"inverse_multiquadric"

    else:
        raise ValueError(f"Unsupported kernel: {kernel_name}")

    return kernel_type, np.array([param], dtype=float)
