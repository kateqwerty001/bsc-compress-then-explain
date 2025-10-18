import os
import sys
import numpy as np
from bonXAI.core.explainer import Explainer
import argparse
from bonXAI.core.utils import compresspp_kt_output_size, set_global_seed
from bonXAI.core.data_loader import DataLoader

sys.stdout.reconfigure(line_buffering=True)

def run_explanations(
    dataset_name,
    X_background,
    X_foreground, 
    y_foreground,
    model,
    model_name, 
    explainer_name, 
    strategy, 
    num_repeats,
    seed,
    n_jobs,
):
    print(f"\n=== Running {explainer_name} ({strategy}) for dataset: {dataset_name} ===")

    explanations = []
    times = []

    for i in range(num_repeats):
        print(f"\n[INFO] Repeat {i + 1}")
        set_global_seed(i)

        explainer = Explainer(
            model=model,
            explainer_name=explainer_name,
            strategy=strategy,
            seed=seed + i
        )

        exp, elapsed_time = explainer.explain(
            X_background=X_background,
            X_foreground=X_foreground,
            y_foreground=y_foreground,
            n_jobs=n_jobs,
        )

        explanations.append(exp)
        times.append(elapsed_time)

    explanations = np.array(explanations)
    times = np.array(times)

    save_dir = f"package_metadata/openml/{dataset_name}/ground_truth"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"shap_kernel_{num_repeats}_{model_name}.npz")

    np.savez_compressed(save_path, exp_values=explanations, times=times)

    print(f"[DONE] Saved explanations and timings to: {save_path}")
    print(f"[DONE] Completed {num_repeats} repeats for {dataset_name}.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--explainer_name", type=str,required=True)
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--n_jobs", type=int, required=True)
    args = parser.parse_args()

    set_global_seed(0)
    loader = DataLoader()
    dataset_name, X_train, y_train, X_test, y_test, model, _ = loader.load_from_openml(dataset_id=int(args.dataset_id), model_name=args.model_name)

    X_test = X_test[:10]
    y_test = y_test[:10]

    print(f"\n[START] Preparing dataset: {dataset_name}")

    model.model_.eval()

    # For SHAP Kernel we use max 4096 samples as foreground points
    if args.explainer_name == "shap" and args.strategy == "kernel" and len(X_test) > 4096:
        rng = np.random.default_rng(int(args.dataset_id)) # seed is set for future comparability with other methods
        ids = rng.choice(len(X_test), size=4096, replace=False)
        X_foreground = X_test[ids]
        y_foreground = y_test[ids]
    else:
        X_foreground = X_test
        y_foreground = y_test

    # For both SHAP Kernel and SAGE Permutation we use max 5000 samples as background points
    if len(X_test) > 5000:
        rng = np.random.default_rng(13 + int(args.dataset_id))
        X_background = rng.choice(X_test, size=5000, replace=False)
    else:
        X_background = X_test
       
    run_explanations( 
        dataset_name = dataset_name,
        X_background = X_background,
        X_foreground = X_foreground, 
        y_foreground = y_foreground,
        model = model,
        model_name = args.model_name, 
        explainer_name = args.explainer_name, 
        strategy = args.strategy, 
        num_repeats = 3,
        seed = 42,
        n_jobs = int(args.n_jobs),
    )
