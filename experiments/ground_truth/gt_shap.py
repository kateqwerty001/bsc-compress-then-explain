import os
import sys
import numpy as np
from bonXAI.core.explainer import Explainer
from bonXAI.core.utils import compresspp_kt_output_size, set_global_seed
from openxai.model import LoadModel, ReturnLoaders

sys.stdout.reconfigure(line_buffering=True)


def run_single_shap(X, y, model, seed, repeat_idx, n_jobs_inner):
    set_global_seed(seed + repeat_idx)

    bg_threshold = None
    if len(X) > 2500:
        bg_threshold = max(compresspp_kt_output_size(X) * 50, 2500)

    print(f"\n[INFO] Repeat {repeat_idx + 1}")

    explainer = Explainer(
        model=model,
        explainer_name="shap",
        strategy="kernel",
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


def run_shap_explanations(
    X,
    y,
    model,
    data_name: str = "custom_dataset",
    num_repeats: int = 3,
    seed: int = 42,
    n_jobs_inner: int = 50,
):
    print(f"\n=== Running SHAP (kernel) for dataset: {data_name} ===")

    explanations = []
    times = []

    for i in range(num_repeats):
        exp, t = run_single_shap(X, y, model, seed, i, n_jobs_inner)
        explanations.append(exp)
        times.append(t)

    explanations = np.array(explanations)
    times = np.array(times)

    save_dir = f"package_metadata/{data_name}/ground_truth"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"explanations_shap_kernel_{num_repeats}_repeats.npz")

    np.savez_compressed(save_path, exp_values=explanations, times=times)

    print(f"[DONE] Saved explanations and timings to: {save_path}")
    print(f"[DONE] Completed {num_repeats} repeats for {data_name}.\n")


if __name__ == "__main__":
    datasets = ["compas", "german", "heart", "gaussian", "heloc", "adult"]

    for name in datasets:
        print(f"\n[START] Preparing dataset: {name}")

        _, loader_test = ReturnLoaders(data_name=name, download=True, batch_size=128)
        X_test = loader_test.dataset.data
        y_test = loader_test.dataset.targets.to_numpy()
        model = LoadModel(data_name=name, ml_model="ann", pretrained=True)
        model.eval()

        run_shap_explanations(
            X=X_test,
            y=y_test,
            model=model,
            data_name=name,
            num_repeats=3,
            seed=0,
            n_jobs_inner=50
        )
