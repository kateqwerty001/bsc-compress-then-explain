import os
import sys
import numpy as np
from bonXAI.core.explainer import Explainer
import argparse
from bonXAI.core.utils import compresspp_kt_output_size, set_global_seed
from bonXAI.core.data_loader import DataLoader
from bonXAI.core.utils import CC18_ALL, CTR23_ALL
from bonXAI.core.preprocessor import Preprocessor

sys.stdout.reconfigure(line_buffering=True)


def run_compression(
    dataset_name,
    X_test,
    y_test,
    model,
    model_name,
    compression_name,
    data_modification_method,
    seed,
    target_size=None,
    m=0,
    kernel='gaussian',
):
    print(f"\n=== Running {compression_name} for dataset: {dataset_name} ===")

    preprocessor = Preprocessor(
        model=model,
        X=X_test,
        y=y_test,
        compression_method=compression_name,
        data_modification_method=data_modification_method,
        seed=seed
    )

    X_compressed, y_compressed, indices, time = preprocessor._preprocess(
        m=m,
        target_size=target_size,
        kernel=kernel
    )

    mod_suffix = (
        f"{data_modification_method}"
        if data_modification_method and str(data_modification_method).lower() != "none"
        else "none"
    )
    size_suffix = (
        f"{target_size}"
        if target_size and str(target_size).lower() != "none"
        else (f"{m}" if m else "none")
    )
    save_dir = f"package_metadata/openml/{dataset_name}/{compression_name}/{mod_suffix}/{size_suffix}/{kernel}/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{compression_name}_{mod_suffix}_{size_suffix}_{kernel}_{model_name}.npz")

    np.savez_compressed(
        save_path,
        X_compressed=X_compressed,
        y_compressed=y_compressed,
        indices=indices,
        compression_time=time,
    )

    print(f"[DONE] Saved compression and timings to: {save_path}")

    return X_compressed, y_compressed

def run_explanations(
    dataset_name,
    X_test,
    y_test,
    X_foreground,
    y_foreground,
    model,
    model_name,
    explainer_name,
    strategy,
    task_type,
    num_repeats,
    seed,
    n_jobs,
    compression_name,
    data_modification_method,
    target_size=None,
    m=0,
    kernel='gaussian',
):
    print(f"\n=== Running {explainer_name} ({strategy}) for dataset: {dataset_name} ===")

    explanations = []
    times = []

    for i in range(num_repeats):
        print(f"\n[INFO] Repeat {i + 1}")

        # --- resample background points each repeat ---
        rng_bg = np.random.default_rng(seed + i)
        if len(X_test) > 5000:
            ids_bg = rng_bg.choice(len(X_test), size=5000, replace=False)
            X_background = X_test[ids_bg]
        else:
            X_background = X_test
        # ------------------------------------------------

        explainer = Explainer(
            model=model,
            explainer_name=explainer_name,
            strategy=strategy,
            task_type=task_type,
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



    mod_suffix = (
        f"{data_modification_method}"
        if data_modification_method and str(data_modification_method).lower() != "none"
        else "none"
    )
    size_suffix = (
        f"{target_size}"
        if target_size and str(target_size).lower() != "none"
        else (f"{m}" if m else "none")
    )
    save_dir = f"package_metadata/openml/{dataset_name}/{compression_name}/{mod_suffix}/{size_suffix}/{kernel}/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{explainer_name}_{strategy}_{num_repeats}_{compression_name}_{mod_suffix}_{size_suffix}_{kernel}_{model_name}.npz")

    np.savez_compressed(save_path, exp_values=explanations, times=times)

    print(f"[DONE] Saved explanations and timings to: {save_path}")
    print(f"[DONE] Completed {num_repeats} repeats for {dataset_name}.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--compression_name", type=str, required=True)
    parser.add_argument("--data_modification_method", type=str, required=False, default=None)
    parser.add_argument("--seed", type=int, required=False, default=0)
    parser.add_argument("--target_size", type=int, required=False, default=None)
    parser.add_argument("--m", type=int, required=False, default=0)
    parser.add_argument("--kernel", type=str, required=False, default='gaussian')
    args = parser.parse_args()

    set_global_seed(args.seed)

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
        or (args.explainer_name == "expected_gradients" and args.strategy == "expected_gradients")
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

    X_compressed, y_compressed = run_compression(
        dataset_name=dataset_name,
        X_test=X_test,
        y_test=y_test,
        model=model,
        model_name=args.model_name,
        compression_name=args.compression_name,
        data_modification_method=args.data_modification_method,
        seed=args.seed,
        target_size=args.target_size,
        m=args.m,
        kernel=args.kernel,
    )

    run_explanations(
        dataset_name=dataset_name,
        X_test=X_compressed,
        y_test=y_compressed,
        X_foreground=X_foreground,
        y_foreground=y_foreground,
        model=model,
        model_name=args.model_name,
        explainer_name=args.explainer_name,
        strategy=args.strategy,
        task_type=task_type,
        num_repeats=3,
        seed=42,
        n_jobs=int(args.n_jobs),
        compression_name=args.compression_name,
        data_modification_method=args.data_modification_method,
        target_size=args.target_size,
        m=args.m,
        kernel=args.kernel,
    )
