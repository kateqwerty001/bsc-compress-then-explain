import os
import random
import numpy as np
import math
import openml
from openxai.model import LoadModel, ReturnLoaders
from sklearn.model_selection import train_test_split
import torch

# CTR23 benchmark suite - regression tasks
CTR23_ALL = [44956, 44957, 44958, 44990, 44977, 44994, 44959, 44984, 44978, 44979, 44960, 45012, 
         44962, 44992, 44965, 44973, 44993, 44980, 44989, 44983, 41021, 44969, 44963, 44981, 
         44970, 44972, 44976, 44987, 44966, 45402, 44967, 44964, 44974, 44975, 44971]
CTR23_ALL = sorted(CTR23_ALL)

# used for experiments 
CTR23_LARGE = [44964, 44975, 44981, 44992]
CTR23_SMALL = [44956, 44963, 44969, 44971, 44973, 44974, 44976, 44977, 44978, 44979, 44980, 44983, 44984, 44989, 44990, 44993, 45012]

# CC18 benchmark suite - classification tasks
# used for experiments 
CC18_LARGE = [28, 44, 182, 300, 554, 1486, 1475, 4538, 1478, 40499, 40668, 40996, 40923, 40927]
CC18_SMALL = [6, 32, 151, 1053, 1590, 1489, 1497, 4534, 1461, 40983, 41027, 23517, 40701]
CC18_ALL = CC18_LARGE + CC18_SMALL

def set_global_seed(seed: int) -> None:
    """
    Sets seed for all known sources of randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["JOBLIB_START_METHOD"] = "spawn"


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


def compresspp_kt_output_size(X: np.ndarray) -> int:
    """
    Returns compressed explanation size for compresspp_kt().
    """
    n = len(X)
    n_prime = 4 ** int(np.floor(np.log(n) / np.log(4)))
    return int(np.sqrt(n_prime))