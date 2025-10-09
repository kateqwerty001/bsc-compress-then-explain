import os
import sys
import numpy as np
import joblib
from bonXAI.core.explainer import Explainer
from bonXAI.core.utils import compresspp_kt_output_size, set_global_seed

sys.stdout.reconfigure(line_buffering=True)


def run_single_sage(X, y, model, seed, repeat_idx, n_jobs_inner):
    set_global_seed(seed + repeat_idx)

    bg_threshold = None
    if len(X) > 2500:
        bg_threshold = min(max(compresspp_kt_output_size(X) * 50, 2500), 5_000)

    print(f"\n[INFO] Repeat {repeat_idx + 1}")

    explainer = Explainer(
        model=model,
        explainer_name="sage",
        strategy="permutation",
        seed=seed + repeat_idx
    )

    exp, elapsed_time = explainer.explain(
        x_background=X,
        x_foreground=X,
        y_foreground=y,
        n_jobs=n_jobs_inner,
        bg_threshold=bg_threshold
    )

    print(f"[INFO] Repeat {repeat_idx + 1} finished in {elapsed_time:.2f}s.")
    return exp, elapsed_time


def run_sage_explanations(
    X,
    y,
    model,
    data_name,
    num_repeats: int = 3,
    seed: int = 42,
    n_jobs_repeats: int = 3,
    n_jobs_inner: int = 16,
):
    print(f"\n=== Running SAGE (permutation) for dataset: {data_name} ===")

    results = joblib.Parallel(n_jobs=n_jobs_repeats)(
        joblib.delayed(run_single_sage)(
            X, y, model, seed, i, n_jobs_inner
        ) for i in range(num_repeats)
    )

    explanations, times = zip(*results)
    explanations = np.array(explanations)
    times = np.array(times)

    save_dir = f"package_metadata/{data_name}/ground_truth"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"explanations_sage_permutation_{num_repeats}_repeats.npz")

    np.savez_compressed(save_path, exp_values=explanations, times=times)

    print(f"[DONE] Saved explanations and timings to: {save_path}")
    print(f"[DONE] Completed {num_repeats} repeats for {data_name}.\n")


if __name__ == "__main__":
    # Example: using openxai loaders for now, but can be replaced by OpenML in the future
    from openxai.model import LoadModel, ReturnLoaders

    datasets = ["compas", "gaussian", "adult", "heart", "heloc", "german"]

    for name in datasets:
        _, loader_test = ReturnLoaders(data_name=name, download=True, batch_size=128)
        X_test = loader_test.dataset.data
        y_test = loader_test.dataset.targets.to_numpy()
        model = LoadModel(data_name=name, ml_model="ann", pretrained=True)
        model.eval()

        run_sage_explanations(
            X=X_test,
            y=y_test,
            model=model,
            data_name=name,
            num_repeats=3,
            seed=0,
            n_jobs_repeats=3,
            n_jobs_inner=16
        )
