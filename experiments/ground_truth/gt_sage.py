import sys
import os
import numpy as np
import joblib
from bonXAI.core.explainer import Explainer
from bonXAI.core.utils import compresspp_kt_output_size, set_global_seed
from openxai.model import LoadModel, ReturnLoaders

sys.stdout.reconfigure(line_buffering=True)


def _single_repeat_sage(data_name, strategy, batch_size, seed, repeat_idx, n_jobs_inner):
    set_global_seed(seed + repeat_idx)

    _, loader_test = ReturnLoaders(data_name=data_name, download=False, batch_size=batch_size)
    X_test = loader_test.dataset.data
    y_test = loader_test.dataset.targets.to_numpy()

    bg_threshold = None
    if len(X_test) > 2500:
        bg_threshold = max(compresspp_kt_output_size(X_test) * 50, 2500)

    model = LoadModel(data_name=data_name, ml_model="ann", pretrained=True)
    model.eval()
    print(f"Repeat {repeat_idx + 1} starting...")
    explainer = Explainer(model=model, explainer_name="sage", strategy="permutation", seed=seed + repeat_idx)
    
    exp, t = explainer.explain(
        x_background=X_test,
        x_foreground=X_test,
        y_foreground=y_test,
        n_jobs=n_jobs_inner,
	bg_threshold=bg_threshold
    )
    print(f"Repeat {repeat_idx + 1} done in {t:.2f}s")
    return exp, t


def run_explanation_sage_parallel(
    data_name: str,
    strategy: str = "permutation",
    num_repeats: int = 3,
    batch_size: int = 128,
    seed: int = 42,
    n_jobs_repeats: int = 3,
    n_jobs_inner: int = 16
):
    results = joblib.Parallel(n_jobs=n_jobs_repeats)(
        joblib.delayed(_single_repeat_sage)(data_name, strategy, batch_size, seed, i, n_jobs_inner)
        for i in range(num_repeats)
    )

    explanations, times = zip(*results)
    explanations = np.array(explanations)
    times = np.array(times)

    save_dir = f"package_metadata/{data_name}/ground_truth"
    os.makedirs(save_dir, exist_ok=True)

    npz_path = os.path.join(
        save_dir,
        f"explanations_sage_{strategy}_{num_repeats}_repeats.npz"
    )
    np.savez_compressed(npz_path, exp_values=explanations, times=times)
    print(f"All explanations and times saved to {npz_path}. {num_repeats} repeats information saved.")


if __name__ == "__main__":
    datasets = ["adult"]

    for data_name in datasets:
        print(f"Running SAGE (permutation) explanations for {data_name}...")
        run_explanation_sage_parallel(
            data_name=data_name,
            strategy="permutation",
            num_repeats=3,
            batch_size=128,
            seed=0,
            n_jobs_repeats=3,   
            n_jobs_inner=16     
        )
