import sys
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import argparse

from bonXAI.core.explainer import Explainer
from bonXAI.core.utils import set_global_seed
from openxai.model import LoadModel, ReturnLoaders

sys.stdout.reconfigure(line_buffering=True)


def _single_repeat(data_name, explainer_name, variant, batch_size, seed, repeat_idx):
    set_global_seed(seed + repeat_idx)

    _, loader_test = ReturnLoaders(data_name=data_name, download=False, batch_size=batch_size)
    X_test = loader_test.dataset.data
    y_test = loader_test.dataset.targets.to_numpy()

    model = LoadModel(data_name=data_name, ml_model="ann", pretrained=True)
    model.eval()

    explainer = Explainer(model=model, name=explainer_name, variant=variant, seed=seed + repeat_idx)
    print(f"Repeat {repeat_idx+1}...")
    exp, t = explainer.explain(X_test, y_test)
    
    if explainer_name.lower() == "shap":
        exp = np.mean(exp, axis=0)
    
    return exp, t


def run_explanation(
    data_name: str,
    explainer_name: str,
    variant: str,
    num_repeats: int = 3,
    batch_size: int = 128,
    seed: int = 42,
    num_workers: int = 3
):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_single_repeat, data_name, explainer_name, variant, batch_size, seed, i)
            for i in range(num_repeats)
        ]
        results = [f.result() for f in futures]

    explanations, times = zip(*results)
    explanations = np.array(explanations)
    times = np.array(times)

    save_dir = f"../package_metadata/{data_name}/ground_truth"
    os.makedirs(save_dir, exist_ok=True)

    npz_path = os.path.join(
        save_dir,
        f"explanations_{explainer_name}_{variant}_{num_repeats}_repeats.npz"
    )
    np.savez_compressed(npz_path, exp_values=explanations, times=times)
    print(f"All explanations and times saved to {npz_path}. {num_repeats} repeats information saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--explainer-name", type=str, required=True)
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--num-repeats", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=3)

    args = parser.parse_args()

    run_explanation(
        data_name=args.data,
        explainer_name=args.explainer_name,
        variant=args.variant,
        num_repeats=args.num_repeats,
        num_workers=args.num_workers
    )

