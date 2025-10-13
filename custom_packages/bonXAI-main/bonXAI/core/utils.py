import os
import random
import numpy as np
import math
import openml

# CTR23 benchmark suite - regression tasks
CTR23 = [44956, 44957, 44958, 44990, 44977, 44994, 44959, 44984, 44978, 44979, 44960, 45012, 
         44962, 44992, 44965, 44973, 44993, 44980, 44989, 44983, 41021, 44969, 44963, 44981, 
         44970, 44972, 44976, 44987, 44966, 45402, 44967, 44964, 44974, 44975, 44971]
CTR23 = sorted(CTR23)

# CC18 benchmark suite - classification tasks
CC18 = openml.study.get_suite('OpenML-CC18')
CC18 = CC18.data

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

def compresspp_kt_output_size(X):
    """
    Returns the size of the compressed explanation for compresspp_kt() function based on the input data X.
    """
    n = len(X)
    n_prime = 4 ** int(np.floor(np.log(n) / np.log(4)))
    return int(np.sqrt(n_prime))


