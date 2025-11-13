import numpy as np
from numpy.random import SeedSequence
from goodpoints import compressc, kt, compress
import math

def largest_power_of_four(n):
    """
    Returns largest power of four less than or equal to n
    """
    return 4**( (n.bit_length() - 1 )//2)

def compute_m(n, target_size, g, num_bins):
    n_prime = num_bins * largest_power_of_four(math.floor(n / num_bins))
    compress_coreset_size = min(2**g * math.sqrt(n_prime * num_bins), n_prime)
    if target_size > compress_coreset_size:
        raise ValueError("target_size is too large for bonxai_compress with set num_bins and g parameters")
    if target_size == compress_coreset_size:
        return 0
    else:
        total_halvings = 1
        while compress_coreset_size / (2**total_halvings) > target_size:
            total_halvings += 1
        return total_halvings

def bonxai_compress(X, kernel_type, k_params=np.ones(1), g=4, num_bins=32,
                target_size=None, delta=0.5, seed=None, mean0=False):
    n = X.shape[0]

    nearest_pow_four = largest_power_of_four(n)
    sqrt_nearest_pow_four = int(math.sqrt(nearest_pow_four))

    if target_size>n or target_size<1:
        raise ValueError("target_size must be in [1, n] and a power of 2")

    if nearest_pow_four != n:
        input_indices = np.linspace(n-1, 0, nearest_pow_four, dtype=int)[::-1]
        return input_indices[ bonxai_compress(
            X[input_indices], kernel_type, k_params=k_params, g=g, 
            num_bins=num_bins, target_size=target_size, delta=delta, seed=seed, mean0=mean0) ]
    
    m = compute_m(n, target_size, g, num_bins)
    log2_target_size = int(math.log2(target_size))

    if sqrt_nearest_pow_four == target_size:
        if log2_target_size <= m:
            print("No compress")
            K = np.empty((n, n))
            compressc.compute_K(X, np.arange(n, dtype=int), kernel_type, k_params, 
                                K)
            
            return kt.thin_K(K, K, log2_target_size, delta=delta, seed=seed, mean0=mean0)
        
    thin_frac = m / (m + (2**m) * (log2_target_size - m))
    thin_delta = delta * thin_frac
    compress_delta = delta * (1-thin_frac)
    
    seed_seqs = SeedSequence(seed).spawn(2)
    compress_seed = seed_seqs[0].generate_state(1)
    thin_seed = seed_seqs[1].generate_state(1)

    compress_coreset = compress.compress_kt(
        X, kernel_type, g=g, num_bins=num_bins, k_params=k_params, 
        delta=compress_delta, seed=compress_seed)
    
    compress_coreset_size = compress_coreset.shape[0]
    K = np.empty((compress_coreset_size, compress_coreset_size))

    compressc.compute_K(X, compress_coreset, kernel_type, k_params, K)
    
    return compress_coreset[ 
        kt.thin_K(K, K, m, delta=thin_delta, seed=thin_seed, mean0=mean0)]
    

    

    
    

    
    