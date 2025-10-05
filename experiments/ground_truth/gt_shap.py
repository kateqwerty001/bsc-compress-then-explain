import sys
import os
import numpy as np
import pandas as pd

from bonXAI.core.explainer import Explainer
from bonXAI.core.utils import compresspp_kt_output_size
from bonXAI.core.utils import set_global_seed
from openxai.model import LoadModel, ReturnLoaders

sys.stdout.reconfigure(line_buffering=True)


def _single_repeat(data_name, explainer_name, strategy, batch_size, seed, repeat_idx):
    set_global_seed(seed + repeat_idx)

    _, loader_test = ReturnLoaders(data_name=data_name, download=False, batch_size=batch_size)
    X_test = loader_test.dataset.data
    y_test = loader_test.dataset.targets.to_numpy()

    bg_theshold = None
    if len(X_test) > 2500:
        bg_theshold = max(compresspp_kt_output_size(X_test)*50, 2500)

    model = LoadModel(data_name=data_name, ml_model="ann", pretrained=True)
    model.eval()
    
    explainer = Explainer(model=model, explainer_name=explainer_name, strategy=strategy, seed=seed + repeat_idx)
    print(f"Repeat {repeat_idx+1}...")
    if explainer_name == "shap" and strategy == "kernel":
        exp, t = explainer.explain(x_background=X_test, x_foreground=X_test, y_foreground=y_test, n_jobs=50, bg_threshold=bg_theshold)
    elif explainer_name == "sage" and strategy == "permutation":
        exp, t = explainer.explain(x_background=X_test, x_foreground=X_test, y_foreground=y_test, n_jobs=16, bg_threshold=bg_theshold)
    return exp, t


def run_explanation(
    data_name: str,
    explainer_name: str,
    strategy: str,
    num_repeats: int = 3,
    batch_size: int = 128,
    seed: int = 42
):
    explanations = []
    times = []

    for i in range(num_repeats):
        exp, t = _single_repeat(data_name, explainer_name, strategy, batch_size, seed, i)
        explanations.append(exp)
        times.append(t)

    explanations = np.array(explanations)
    times = np.array(times)

    save_dir = f"package_metadata/{data_name}/ground_truth"
    os.makedirs(save_dir, exist_ok=True)

    npz_path = os.path.join(
        save_dir,
        f"explanations_{explainer_name}_{strategy}_{num_repeats}_repeats.npz"
    )
    np.savez_compressed(npz_path, exp_values=explanations, times=times)
    print(f"All explanations and times saved to {npz_path}. {num_repeats} repeats information saved.")


if __name__ == "__main__":
    datasets = ["compas", "german", "heart", "gaussian", "heloc", "adult"]

    for data_name in datasets:
        for explainer, strategy in [("shap", "kernel")]:
            print(f"Running explanations for {data_name} using {explainer} ({strategy})...")
            run_explanation(
                data_name=data_name,
                explainer_name=explainer,
                strategy=strategy,
                num_repeats=3,
                batch_size=128,
                seed=0
            )
