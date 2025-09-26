import os
import random
import numpy as np
import math

def set_global_seed(seed: int) -> None:
    """
    Sets seed for all known sources of randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def possible_g_values(n_samples: int, num_bins: int) -> list[int]:
    """
    Returns a list of possible values of g for compress_kt() function based on the number of samples and bins.
    """
    x = n_samples // num_bins
    power = 1
    while power * 4 <= x:
        power *= 4
    n_prime = num_bins * power

    g = 0
    possible_g = []
    while (2 ** g) * math.sqrt(n_prime * num_bins) <= n_prime:
        possible_g.append(g)
        g += 1
    return list(reversed(possible_g))

def possible_num_bins_values(n_samples: int) -> list[int]:
    """
    Returns a list of possible values of num_bins for compress_kt() function based on the number of samples.
    """
    possible_num_bins = []
    power = 1 # assume that num_bins starts at 4
    while (val := 4 ** power) <= n_samples:
        possible_num_bins.append(val)
        power += 1
    return possible_num_bins


