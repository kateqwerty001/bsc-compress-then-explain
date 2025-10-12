import numpy as np
import pandas as pd
import os
import time
import sys
from bonXAI.core.preprocessor import Preprocessor
from bonXAI.core.explainer import Explainer
from bonXAI.core.evaluation import Evaluator
from bonXAI.core.utils import set_global_seed
from openxai.model import LoadModel, ReturnLoaders
import argparse

sys.stdout.reconfigure(line_buffering=True)

kernels = ["gaussian", "sobolev", "inverse_multiquadric",
            "matern_0.5", "matern_1.5", "matern_2.5"]


def run_pipeline(dataset_name, X, y, model, explainer_name, strategy, n_jobs):
    SEED = 0
    N_REPEATS = 15
    set_global_seed(SEED)

    gt = np.load(f"/mnt/evafs/faculty/home/kbokhan/bsc-compress-then-explain/package_metadata/{dataset_name}/ground_truth/explanations_{explainer_name}_{strategy}_3_repeats.npz")
    gt_exp_values, gt_times = gt["exp_values"], gt["times"]

    mean_gt_exp_values = np.mean(gt_exp_values, axis=0) # Average over repeats

    evaluator = Evaluator(ground_truth_explanation=mean_gt_exp_values, ground_truth_points=X)

    results = []

    print("\n[INFO] Running Kernel Thinning for different kernels...")
    for i in range(N_REPEATS):
        prev_size = -1
        for kernel in kernels:
            pre = Preprocessor(
                X=X,
                y=y,
                model=model,
                compression_method="kernel_thinning",
                data_modification_method="none",
                seed=SEED + i
            )
            counter = 3
            for m in range(15, -10, -1):
                print(f"Repeat {i+1}/{N_REPEATS}, Kernel: {kernel}, m={m}")
                if  counter <= 0:
                    continue

                try:
                    X_kt, y_kt, idx_kt, t_kt = pre._preprocess(
                        g=4,
                        num_bins=32,
                        m=m,
                        kernel=kernel
                    )
                    
                except Exception as e:
                    continue

                if len(X_kt) == prev_size:
                    continue

                start = time.time()
                explainer = Explainer(
                    model=model,
                    explainer_name=explainer_name,
                    strategy=strategy,
                    seed=SEED + i
                )

                exp_values, time_ = explainer.explain(x_foreground=X, x_background=X_kt, y_foreground=y, n_jobs=n_jobs)

                row = evaluator.evaluate_explanation(exp_values, time_, len(X_kt))
                row.update(evaluator.evaluate_compression(X_kt))
                row.update({
                    "explainer": explainer_name,
                    "strategy": strategy,
                    "method": "kernel_thinning",
                    "kernel": kernel,
                    "g": 4,
                    "num_bins": 32,
                    "m": m,
                    "compression_time": t_kt,
                    "explanation_time": time.time() - start,

                })
                results.append(row)
                prev_size = len(X_kt)
                counter-=1

    print("\n[INFO] Running IID Baseline...")
    sizes = pd.DataFrame(results)["size"].unique()
    for i in sizes:
        for j in range(N_REPEATS):
            pre = Preprocessor(
                X=X,
                y=y,
                model=model,
                compression_method="iid",
                data_modification_method="none",
                seed=SEED + j + i
            )
            X_iid, y_iid, idx_iid, t_iid = pre._preprocess(target_size=i)
            start = time.time()
            explainer = Explainer(
                model=model,
                explainer_name=explainer_name,
                strategy=strategy,
                seed=SEED + j + i
            )
            exp_values, time_ = explainer.explain(x_foreground=X, x_background=X_iid, y_foreground=y, n_jobs=n_jobs)
            
            row = evaluator.evaluate_explanation(exp_values, time_, len(X_iid))
            row.update(evaluator.evaluate_compression(X_iid))
            row.update({
                "explainer": explainer_name,
                "strategy": strategy,
                "method": "iid",
                "kernel": "none",
                "g": None,
                "num_bins": None,
                "m": None,
                "compression_time": t_iid,
                "explanation_time": time.time() - start,
            })
            results.append(row)

    df = pd.DataFrame(results)
    save_dir = f"results/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"results_{explainer_name}_{strategy}_kernels_comparison.csv")

    df.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--explainer_name", type=str,required=True)
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--n_jobs", type=int, required=True)
    args = parser.parse_args()

    dataset_name = args.dataset
    explainer_name = args.explainer_name
    strategy = args.strategy
    n_jobs = args.n_jobs

    print(f"\n[START] Preparing dataset: {dataset_name}")

    _, loader_test = ReturnLoaders(data_name=dataset_name, download=True, batch_size=128)
    X_test = loader_test.dataset.data
    y_test = loader_test.dataset.targets.to_numpy()
    model = LoadModel(data_name=dataset_name, ml_model="ann", pretrained=True)
    model.eval()

    print(f"[INFO] Starting {explainer_name}-{strategy} for dataset: {dataset_name}")
    run_pipeline(
        dataset_name=dataset_name,
        X=X_test, 
        y=y_test,
        model=model,
        explainer_name=explainer_name,
        strategy=strategy,
        n_jobs=n_jobs,
    )
