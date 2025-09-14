
import numpy as np
from numpy.random import SeedSequence
from goodpoints import compressc, kt, compress

def bonxai_compress(X, kernel_type, k_params=np.ones(1), g=0, num_bins=4,
                   m=0, delta=0.5, seed=None, mean0=False):
    """Returns coreset of size floor(sqrt(n')*(2^(m))) if m is not None, or 
    the coreset size 2^g * sqrt(n * num_bins), as row indices into X. 
    Here n' = the largest power of 4 less than or equal to n. 
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      kernel_type: Byte string name of kernel to use:
        b"gaussian" for sum-of-Gaussians kernel 
        b"sobolev" for sum-of-Sobolevs kernel
        b"inverse_multiquadric" for sum-of-inverse-multiquadrics kernel
      k_params: Double array of kernel parameters
      g: Oversampling parameter, a nonnegative integer
      num_bins: Number of bins, a power of 4 <= n
      m: Multiplicative factor for target size, integer; 
        to compress to the size of sqrt(n') set m=0; to return the result of oversampling compression set m=None
      delta: Failure probability parameter for kernel thinning
      seed: Nonnegative integer seed to initialize a random number 
        generator or None to set no seed
      mean0: If False, final KT call minimizes MMD to empirical measure over 
        the input points. Otherwise minimizes MMD to the 0 measure; this
        is useful when the kernel has expectation zero under a target measure.
    """
    n = X.shape[0]
    nearest_pow_four = compress.largest_power_of_four(n)

    if nearest_pow_four != n and m is not None:
        input_indices = np.linspace(n-1, 0, nearest_pow_four, dtype=int)[::-1]
        return input_indices[ bonxai_compress(
            X[input_indices], kernel_type, k_params=k_params, g=g, 
            num_bins=num_bins, m=m, delta=delta, seed=seed, mean0=mean0) ]
    
    if m is None: 
        # no Thin step needed, return the compress_kt coreset of size 2^g * sqrt(n * num_bins)
        return compress.compress_kt(
        X, kernel_type, g=g, num_bins=num_bins, k_params=k_params, 
        delta=delta, seed=seed)
    
    m = g + (num_bins.bit_length() - 1)//2 + m
    
    # If 2^m >= sqrt(n) or equivalently 0 >= log2(sqrt(n)) - m, then 
    # no Compress step is needed: Thin directly to size sqrt(n)
    # Note: (n.bit_length() - 1)//2 = log2(sqrt(n))
    log2_sqrtn = (n.bit_length()-1)//2
    if log2_sqrtn <= m:
        # Allocate memory for kernel matrix
        K = np.empty((n, n))
        # Compute compress_coreset kernel matrix in place
        compressc.compute_K(X, np.arange(n, dtype=int), kernel_type, k_params, 
                            K)
        return kt.thin_K(K, K, log2_sqrtn, delta=delta, seed=seed, mean0=mean0)
        
    # Otherwise, divide failure probability between Compress and Thin rounds
    thin_frac = m / (m + (2**m) * (log2_sqrtn - m))
    thin_delta = delta * thin_frac
    compress_delta = delta * (1-thin_frac)
    
    # Generate one seed for the Compress step and one for the final Thin step
    seed_seqs = SeedSequence(seed).spawn(2)
    compress_seed = seed_seqs[0].generate_state(1)
    thin_seed = seed_seqs[1].generate_state(1)
    
    #
    # Compress step
    #
    # Break input into num_bins bins, use Compress(g) to create a coreset of size 
    # 2^g sqrt(n / num_bins) for each bin, and concatenate bin coresets
    compress_coreset = compress.compress_kt(
        X, kernel_type, g=g, num_bins=num_bins, k_params=k_params, 
        delta=compress_delta, seed=compress_seed)
    
    #
    # Thin step
    #
    # Allocate memory for kernel matrix
    compress_coreset_size = compress_coreset.shape[0]
    K = np.empty((compress_coreset_size, compress_coreset_size))
    # Compute compress_coreset kernel matrix in place
    compressc.compute_K(X, compress_coreset, kernel_type, k_params, K)
    # Use target kt.thin to reduce coreset size from 2^g * sqrt(n * num_bins)
    # to sqrt(n)
    return compress_coreset[ 
        kt.thin_K(K, K, m, delta=thin_delta, seed=thin_seed, mean0=mean0)]