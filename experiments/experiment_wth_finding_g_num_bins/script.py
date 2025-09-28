import time
import numpy as np
import os
import sys
import pandas as pd
from sklearn.datasets import make_blobs
from bonXAI.core.bonxai_compress import bonxai_compress
from bonXAI.core.preprocessor import Preprocessor, Compressor
from bonXAI.core.metrics import compute_mmd 
from openxai.dataloader import ReturnLoaders

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


sys.stdout.reconfigure(line_buffering=True)

SEED = 42

def generate_data_no_label(dataset_name, records_num, features_num = None, batch_size=128):
    if dataset_name == "blobs":
        X, y = make_blobs(n_samples=records_num, n_features=features_num, centers=100, 
                 cluster_std=0.5, random_state=SEED)
        df_data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        print(df_data.head())
        return df_data
    
    elif dataset_name=="gaussian" or dataset_name=="compas" or dataset_name=="german" or dataset_name=="heart" or dataset_name=="adult" or dataset_name=="heloc":
        _, loader_test = ReturnLoaders(data_name=dataset_name, download=False, batch_size=batch_size)
        X = loader_test.dataset.data
        y = loader_test.dataset.targets.to_numpy()
        compr = Compressor(X = X, y=y, model=None, seed=42)
        new_X, new_y, _, _ = compr._arfpy_compression(records_num)
        df_data = pd.DataFrame(new_X, columns=[f"feature_{i}" for i in range(new_X.shape[1])])
        return df_data

################################### Experiment Parameters #########################################################################

# dataset selection
dataset = ["gaussian", "compas", "german", "heart", "adult", "heloc", "blobs"]
records_num = [1000, 3_000, 10_000, 20_000]
features_num = [10, 50, 100, 1_000]  # only for blobs dataset

# selection of Kernel Thinning parameters
kernels = [b"inverse_multiquadric", b"sobolev"]
possible_g = [0, 1, 2, 3, 4, 5, 6, 7]
possible_num_bins = [1, 4, 16, 32, 64, 128, 256]

################################### Experiment  ##################################################################################
def run_experiment(dataset_name, records_num, kernel, features_num=None):
    X_df = generate_data_no_label(dataset_name=dataset_name, records_num=records_num, features_num=features_num)
    results = []

    print("-" * 60)
    print("Starting with dataset:", dataset_name)
    print(f"Data shape: {X_df.shape}")
    print(f"Using kernel: {kernel.decode()}")

    X_array = X_df.values if isinstance(X_df, pd.DataFrame) else X_df

    for i, num_bins in enumerate(possible_num_bins):
        for j, g in enumerate(possible_g):
            print(f"Processing: num_bins={num_bins}, g={g} "
                f"({i*len(possible_g) + j + 1}/{len(possible_num_bins)*len(possible_g)})")
            
            try:
                start_time = time.time()
                
                compressed_indices = np.unique(bonxai_compress(X_array, kernel, g=g, num_bins=num_bins, m=None))
                compressed_data = X_array[compressed_indices]
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                compressed_size = compressed_data.shape[0]
                
                mmd_score = compute_mmd(X_array, compressed_data)
                
                result = {
                    'g': g,
                    'num_bins': num_bins,
                    'execution_time_sec': execution_time,
                    'compressed_size': compressed_size,
                    'mmd_score': mmd_score,
                    'success': True
                }
                
                results.append(result)
                print(f"  ✓ Success: time={execution_time:.2f}s, size={compressed_size}, "
                    f"MMD={mmd_score:.6f}")
                
            except Exception as e:
                result = {
                    'g': g,
                    'num_bins': num_bins,
                    'execution_time_sec': None,
                    'compressed_size': None,
                    'mmd_score': None,
                    'success': False,
                    'error': str(e)
                }
                results.append(result)
                print(f"  ✗ Error: {e}")
            
            print("-" * 60)

    df_results = pd.DataFrame(results)
    output_dir = f"experiments/experiment_wth_finding_g_num_bins/results/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}_{records_num}_{features_num}_{kernel.decode()}.csv")
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    print("-" * 60)


if __name__ == "__main__":
    for dataset_name in dataset:
        for rec_num in records_num:
            for kernel in kernels:
                if dataset_name == "blobs":
                    for feat_num in features_num:
                        run_experiment(dataset_name=dataset_name, records_num=rec_num, kernel=kernel, features_num=feat_num)
                else:
                    run_experiment(dataset_name=dataset_name, records_num=rec_num, kernel=kernel)
