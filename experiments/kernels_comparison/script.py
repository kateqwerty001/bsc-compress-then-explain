import numpy as np
import pandas as pd
import os
import argparse
from bonXAI.core.preprocessor import Preprocessor
from bonXAI.core.explainer import Explainer
from bonXAI.core.evaluation import Evaluator
from bonXAI.core.utils import set_global_seed
from openxai.dataloader import ReturnLoaders
from openxai import LoadModel


def run_pipeline(dataset_name, explainer_name, variant):
    SEED = 0
    N_REPEATS = 33
    set_global_seed(SEED)

    kernels = ["gaussian", "sobolev", "inverse_multiquadric",
               "matern_0.5", "matern_1.5", "matern_2.5"]

    _, loader_test = ReturnLoaders(data_name=dataset_name, download=False, batch_size=128)
    X = loader_test.dataset.data
    y = loader_test.dataset.targets.to_numpy()

    model = LoadModel(data_name=dataset_name, ml_model="ann", pretrained=True)

    gt = np.load(f"package_metadata/{dataset_name}/ground_truth/explanations_{explainer_name}_{variant}_3_repeats.npz")
    gt_exp_values, gt_times = gt["exp_values"], gt["times"]
    gt_exp_mean = np.mean(gt_exp_values, axis=0)

    evaluator = Evaluator(ground_truth_explanation=gt_exp_mean, reference_points=X)

    results = []

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
            for m in range(-6, 20, 1):
                print(f"Repeat {i+1}/{N_REPEATS}, Kernel: {kernel}, m={m}")
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

                explainer = Explainer(
                    model=model,
                    name=explainer_name,
                    variant=variant,
                    seed=SEED + i
                )
                exp_values, time = explainer.explain(X=X_kt, y=y_kt)

                if explainer_name == "shap":
                    exp_values = np.mean(exp_values, axis=0)

                row = evaluator.evaluate_explanation(exp_values, time, len(X_kt))
                row.update(evaluator.evaluate_compression(X_kt))
                row.update({
                    "method": "kernel_thinning",
                    "g": 4,
                    "num_bins": 32,
                    "m": m,
                    "compression_time": t_kt,
                    "kernel": kernel,
                    "explainer": explainer_name,
                    "variant": variant,
                })
                results.append(row)
                prev_size = len(X_kt)

    df = pd.DataFrame(results)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "results", dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(
        os.path.join(out_dir, f"results_{explainer_name}_{variant}_kernels_comparison.csv"),
        index=False
    )
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Dataset name")
    parser.add_argument("--explainer-name", type=str, required=True, help="Explainer name")
    parser.add_argument("--variant", type=str, required=True, help="Explainer variant")

    args = parser.parse_args()

    run_pipeline(
        dataset_name=args.data,
        explainer_name=args.explainer_name,
        variant=args.variant
    )
