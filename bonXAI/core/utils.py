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

def possible_num_bins_values(n: int) -> list[int]:
    return [i for i in range(1, n // 4) if n // i > 16]

