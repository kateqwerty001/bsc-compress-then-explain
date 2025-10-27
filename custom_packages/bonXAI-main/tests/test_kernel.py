import numpy as np
import pytest
from bonXAI.core import kernel

def test_gaussian_kernel_params():
    X = np.zeros((10, 5))
    ktype, params = kernel.resolve_kernel_params("gaussian", X)
    assert ktype == b"gaussian"
    # sigma^2 = 2 * num_features = 10
    assert np.allclose(params, [10.0])
    assert params.dtype == float

@pytest.mark.parametrize("name,expected_type,expected_params", [
    ("sobolev", b"sobolev", [1.0]),
    ("inverse_multiquadric", b"inverse_multiquadric", [1.0]),
])
def test_simple_kernels(name, expected_type, expected_params):
    X = np.zeros((8, 3))
    ktype, params = kernel.resolve_kernel_params(name, X)
    assert ktype == expected_type
    assert np.allclose(params, expected_params)
    assert params.ndim == 1

@pytest.mark.parametrize("name,nu", [
    ("matern_0.5", 0.5),
    ("matern_1.5", 1.5),
    ("matern_2.5", 2.5),
])
def test_matern_kernels(name, nu):
    X = np.zeros((4, 2))
    ktype, params = kernel.resolve_kernel_params(name, X)
    assert ktype == b"matern"
    # matern kernels return [1.0, 1.0, nu]
    assert np.allclose(params, [1.0, 1.0, nu])
    assert params.dtype == np.float64

def test_unknown_kernel_raises():
    X = np.zeros((2, 2))
    with pytest.raises(ValueError):
        kernel.resolve_kernel_params("nonexistent", X)

@pytest.mark.parametrize("d", [1, 2, 7, 32])
def test_gaussian_sigma_scales_with_features(d):
    X = np.zeros((5, d))
    _, p = kernel.resolve_kernel_params("gaussian", X)
    assert p.shape == (1,)
    assert p[0] == pytest.approx(2.0 * d)

@pytest.mark.parametrize("name", ["gaussian","sobolev","inverse_multiquadric"])
def test_all_params_are_float64(name):
    X = np.zeros((3, 4))
    _, p = kernel.resolve_kernel_params(name, X)
    assert p.dtype == np.float64

@pytest.mark.parametrize("name", ["gaussian","sobolev","inverse_multiquadric"])
def test_all_params_are_float64(name):
    X = np.zeros((3, 4))
    _, p = kernel.resolve_kernel_params(name, X)
    assert p.dtype == np.float64
