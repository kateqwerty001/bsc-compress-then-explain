import numpy as np
import pytest
from bonXAI.core import bonxai_compress
import math

@pytest.fixture(autouse=True)
def patch_goodpoints(monkeypatch):
    """Patch heavy external functions from goodpoints with fast stubs."""
    called = {"compress": [], "compute_K": [], "thin_K": []}

    def fake_compress_kt(X, kernel_type, **kwargs):
        called["compress"].append(True)
        # return half indices deterministically
        return np.arange(0, X.shape[0] // 2, dtype=int)

    def fake_compute_K(X, idx, kernel_type, k_params, K):
        called["compute_K"].append(True)
        # fill K deterministically
        K[:] = np.eye(len(idx))

    def fake_thin_K(K1, K2, m, **kwargs):
        called["thin_K"].append(True)
        # return every other index (simulate thinning)
        n = K1.shape[0]
        return np.arange(0, max(1, n // 2), dtype=int)

    monkeypatch.setattr(bonxai_compress.compress, "compress_kt", fake_compress_kt)
    monkeypatch.setattr(bonxai_compress.compressc, "compute_K", fake_compute_K)
    monkeypatch.setattr(bonxai_compress.kt, "thin_K", fake_thin_K)

    yield called

def test_basic_compress_none_m_triggers_compress_only(patch_goodpoints):
    X = np.random.randn(16, 3)
    kernel_type = b"gaussian"
    res = bonxai_compress.bonxai_compress(X, kernel_type, m=None)
    # mock compress_kt returns half indices
    assert np.all(res < X.shape[0])
    assert patch_goodpoints["compress"]

def prev_power_of_4(n):
    p = 1
    while p * 4 <= n:
        p *= 4
    return p

@pytest.mark.parametrize("n,m", [(1,0),(3,0),(4,0),(16,1),(64,2)])
def test_size_when_m_specified_is_sqrt_prev_pow4_times_2_pow_m(n, m):
    X = np.random.randn(n, 2)
    res = bonxai_compress.bonxai_compress(X, b"gaussian", m=m)
    expected = int(math.sqrt(prev_power_of_4(n)) * (2 ** m))
    assert res.ndim == 1
    assert len(res) == expected
    assert np.all(res >= 0) and np.all(res < n)
    assert len(np.unique(res)) == len(res)  # no dups

def test_m_none_returns_compress_only_half_indices(patch_goodpoints):
    X = np.random.randn(33, 5)
    res = bonxai_compress.bonxai_compress(X, b"gaussian", m=None, g=2, num_bins=4)
    assert len(res) == X.shape[0] // 2
    assert patch_goodpoints["compress"]
    assert not patch_goodpoints["compute_K"]
    assert not patch_goodpoints["thin_K"]

def test_non_power_of_four_triggers_recursive_call(monkeypatch):
    X = np.random.randn(10, 2)  # not a power of 4
    call_count = {"n": 0}

    real = bonxai_compress.bonxai_compress
    def wrapper(*args, **kwargs):
        call_count["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(bonxai_compress, "bonxai_compress", wrapper)
    bonxai_compress.bonxai_compress(X, b"sobolev", m=0)
    assert call_count["n"] >= 2

def test_thin_branch_triggers_compute_and_thin(patch_goodpoints):
    X = np.random.randn(16, 2)
    kernel_type = b"gaussian"
    # choose m high enough that log2(sqrt(n)) <= m
    res = bonxai_compress.bonxai_compress(X, kernel_type, m=10)
    assert patch_goodpoints["compute_K"]
    assert patch_goodpoints["thin_K"]
    assert isinstance(res, np.ndarray)

def test_compress_then_thin_branch(patch_goodpoints):
    X = np.random.randn(64, 2)
    kernel_type = b"gaussian"
    # m smaller than sqrt(n) threshold
    res = bonxai_compress.bonxai_compress(X, kernel_type, m=1)
    # compress and thin should both be called
    assert patch_goodpoints["compress"]
    assert patch_goodpoints["compute_K"]
    assert patch_goodpoints["thin_K"]
    assert isinstance(res, np.ndarray)
    assert res.ndim == 1

def test_invalid_kernel_type_pass_through():
    X = np.random.randn(8, 3)
    res = bonxai_compress.bonxai_compress(X, b"invalid_kernel", m=None)
    assert isinstance(res, np.ndarray) 

def test_seed_handling_produces_deterministic_results(monkeypatch):
    """Ensure same seed yields same result."""
    X = np.random.randn(16, 2)
    kernel_type = b"gaussian"

    # deterministic fake compress_kt that depends on seed
    def fake_compress_kt(X, kernel_type, **kwargs):
        seed_val = kwargs.get("seed")
        np.random.seed(seed_val[0] if isinstance(seed_val, np.ndarray) else seed_val or 0)
        return np.random.choice(X.shape[0], size=X.shape[0] // 2, replace=False)

    monkeypatch.setattr(bonxai_compress.compress, "compress_kt", fake_compress_kt)

    res1 = bonxai_compress.bonxai_compress(X, kernel_type, m=1, seed=42)
    res2 = bonxai_compress.bonxai_compress(X, kernel_type, m=1, seed=42)
    assert np.array_equal(res1, res2)

def test_negative_or_non_int_g_raises():
    X = np.random.randn(16, 3)
    with pytest.raises((ValueError, AssertionError, TypeError)):
        bonxai_compress.bonxai_compress(X, b"gaussian", m=None, g=-1, num_bins=4)
    with pytest.raises((ValueError, AssertionError, TypeError)):
        bonxai_compress.bonxai_compress(X, b"gaussian", m=None, g=1.5, num_bins=4)

def test_negative_num_bins_raises():
    X = np.random.randn(16, 3)
    with pytest.raises((ValueError, AssertionError, TypeError)):
        bonxai_compress.bonxai_compress(X, b"gaussian", m=None, g=1, num_bins=-4)

def test_tiny_n_behaviour():
    for n in [1, 2, 3]:
        X = np.random.randn(n, 2)
        res = bonxai_compress.bonxai_compress(X, b"gaussian", m=0)
        # sqrt(prev_pow4(1..3)) == 1
        assert len(res) == 1
        assert 0 <= res[0] < n
