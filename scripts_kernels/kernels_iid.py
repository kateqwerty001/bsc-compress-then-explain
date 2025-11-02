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
from bonXAI.core.data_loader import DataLoader
from bonXAI.core.utils import CC18_ALL, CTR23_ALL
import argparse
import hashlib

sys.stdout.reconfigure(line_buffering=True)

kernels = ["gaussian", "sobolev", "inverse_multiquadric",
            "matern_0.5", "matern_1.5", "matern_2.5"]


def run_pipeline_with_kernels_and_iid(dataset_name, X_test, y_test, X_foreground, y_foreground, model, model_name, explainer_name, strategy, task_type, seed, n_jobs):
    N_REPEATS = 20
    set_global_seed(seed)

    gt = np.load(f"/mnt/evafs/faculty/home/kbokhan/bsc-compress-then-explain/package_metadata/{dataset_name}/ground_truth/explanations_{explainer_name}_{strategy}_3_repeats.npz")
    gt_exp_values, gt_times = gt["exp_values"], gt["times"]

    mean_gt_exp_values = np.mean(gt_exp_values, axis=0) # Average over repeats

    evaluator = Evaluator(ground_truth_explanation=mean_gt_exp_values, ground_truth_points=X_test.copy())

    results = []

    print("\n[INFO] Running Kernel Thinning for different kernels...")
    for i in range(N_REPEATS):
        prev_size = -1
        for kernel in kernels:
            counter = 3
            for m in range(15, -10, -1):
                pre = Preprocessor(
                    X=X_test.copy(),
                    y=y_test.copy(),
                    model=model,
                    compression_method="kernel_thinning",
                    data_modification_method="none",
                    seed=seed + i + np.abs(m) + int(hashlib.sha256(kernel.encode()).hexdigest(), 16) % (10**6)
                )
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
                    seed=seed + i + np.abs(m) + int(hashlib.sha256(kernel.encode()).hexdigest(), 16) % (10**6)
                )

                exp_values, time_ = explainer.explain(X_foreground=X_foreground, X_background=X_kt, y_foreground=y_foreground, n_jobs=n_jobs)

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
                X=X_test.copy(),
                y=y_test.copy(),
                model=model,
                compression_method="iid",
                data_modification_method="none",
                seed=seed + j + i
            )
            X_iid, y_iid, idx_iid, t_iid = pre._preprocess(target_size=i)
            start = time.time()
            explainer = Explainer(
                model=model,
                explainer_name=explainer_name,
                strategy=strategy,
                seed=seed + j + i
            )
            exp_values, time_ = explainer.explain(X_foreground=X_foreground, X_background=X_iid, y_foreground=y_foreground, n_jobs=n_jobs)
            
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
    save_dir = f"/mnt/evafs/faculty/home/kbokhan/bsc-compress-then-explain/package_metadata/{dataset_name}/kernels_iid_comparison/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"results_{explainer_name}_{strategy}_kernels_and_iid_comparison.csv")

    df.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--explainer_name", type=str, required=True)
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--n_jobs", type=int, required=True)
    args = parser.parse_args()

    set_global_seed(0)

    loader = DataLoader()
    dataset_name, X_train, y_train, X_test, y_test, model, _ = loader.load_from_openml(
        dataset_id=int(args.dataset_id),
        model_name=args.model_name
    )

    print(f"\n[START] Preparing dataset: {dataset_name}")

    if args.model_name == "ann":
        model.model_.eval()

    # --- select fixed foreground points ---
    if (
        (args.explainer_name == "shap" and args.strategy == "kernel")
        or (args.explainer_name == "expected_gradients" and args.strategy == "na")
    ) and len(X_test) > 4096:
        rng_fg = np.random.default_rng(int(args.dataset_id))
        ids_fg = rng_fg.choice(len(X_test), size=4096, replace=False)
        X_foreground = X_test[ids_fg]
        y_foreground = y_test[ids_fg]
    else:
        X_foreground = X_test
        y_foreground = y_test
    # ---------------------------------------

    # determine task type
    if int(args.dataset_id) in CC18_ALL:
        task_type = "classification"
    elif int(args.dataset_id) in CTR23_ALL:
        task_type = "regression"
    else:
        raise ValueError(f"Dataset ID {args.dataset_id} not found in CC18 or CTR23 benchmarks.")

    print(f"[INFO] Starting {args.explainer_name}-{args.strategy} for dataset: {dataset_name}")
    run_pipeline_with_kernels_and_iid(
        dataset_name=dataset_name,
        X_test=X_test, 
        y_test=y_test,
        X_foreground=X_foreground,
        y_foreground=y_foreground,
        model=model,
        model_name=args.model_name,
        explainer_name=args.explainer_name,
        strategy=args.strategy,
        task_type=task_type,
        seed=42,
        n_jobs=int(args.n_jobs),
    )
