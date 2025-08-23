import sys
import os
import numpy as np
import pandas as pd

from bonXAI.core.explainer import Explainer
from bonXAI.core.utils import set_global_seed
from openxai.model import LoadModel, ReturnLoaders

sys.stdout.reconfigure(line_buffering=True)

def run_explanation(
    data_name: str,
    explainer_name: str,
    variant: str,
    num_repeats: int = 3,
    batch_size: int = 128,
    seed: int = 42
):
    """
    A function used to calculate and save ground truth explanations for a given dataset and explainer.
    """
    set_global_seed(seed)

    _, loader_test = ReturnLoaders(data_name=data_name, download=False, batch_size=batch_size)
    X_test = loader_test.dataset.data
    y_test = loader_test.dataset.targets.to_numpy()

    model = LoadModel(data_name=data_name, ml_model='ann', pretrained=True)
    model.eval()

    explanations = []
    times = []

    for i in range(num_repeats):
        explainer = Explainer(model=model, name=explainer_name, variant=variant, seed=seed+i)
        print(f"Repeat {i+1}/{num_repeats}...")
        exp, t = explainer.explain(X_test, y_test)
        explanations.append(exp)
        times.append(t)

    explanations = np.array(explanations)
    times = np.array(times)

    save_dir = f"package_metadata/{data_name}/ground_truth"
    os.makedirs(save_dir, exist_ok=True)

    npz_path = os.path.join(save_dir, f"explanations_{explainer_name}_{variant}_{num_repeats}_repeats.npz")
    np.savez_compressed(npz_path, exp_values=explanations, times=times)
    print(f"All explanations and times saved to {npz_path}. {num_repeats} repeats information saved.")


run_explanation(
    data_name="adult",
    explainer_name="shap",
    variant="kernel",
    num_repeats=3
)
