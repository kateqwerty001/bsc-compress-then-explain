import numpy as np
import sys
import os
import pandas as pd
from bonXAI.core.preprocessing import Preprocessor
from bonXAI.core.explainer import Explainer
from bonXAI.core.evaluation import Evaluator
from bonXAI.core.utils import set_global_seed
from openxai.model import LoadModel, ReturnLoaders

sys.stdout.reconfigure(line_buffering=True)
sys.path.append("..")

def run_experiment(
    data_name: str,
    methods: list,
    explainer_name: str,
    variant: str,
    kernel: str = "gaussian",
    num_repeats: int = 3,
    batch_size: int = 128,
    seed: int = 42
):
    """
    A function to run an experiment with to compare different methods. Compresspp_kt() is used as the compression methods.
    """
    results = []
    set_global_seed(seed)

    # load the data
    _, loader_test = ReturnLoaders(data_name=data_name, download=False, batch_size=batch_size)

    X = loader_test.dataset.data
    y = loader_test.dataset.targets.to_numpy()

    # load the model
    model = LoadModel(data_name=data_name, ml_model="ann", pretrained=True)
    model.eval()

    # load ground truth values
    ground_truth = np.load(f"package_metadata/{data_name}/ground_truth/explanations_{explainer_name}_{variant}_3_repeats.npz")
    if explainer_name == "shap":
        gt_values_mean = ground_truth['exp_values'].mean(axis=0).mean(axis=0)
    elif explainer_name == "sage":
        gt_values_mean = ground_truth['exp_values'].mean(axis=0)
    else: 
        raise ValueError(f"Explainer {explainer_name} not supported.")
    
    evaluator = Evaluator(ground_truth_explanation=gt_values_mean, reference_points=X)

    for i in range(num_repeats):
        print(f" ----- Running repeat {i+1}/{num_repeats}...")

        for method in methods:
            print(f"Running method {method}...")
            if method == "iid":
                pre = Preprocessor(method=method, model=model, seed=seed+i)
                X_red, y_red, idx, t_comp = pre.run(X, y)
                explainer = Explainer(model=model, name=explainer_name, variant=variant, seed=seed+i)
                values, t_exp = explainer.explain(X_red, y_red)
                exp_mean = values
                if exp_mean.ndim > 1:
                    exp_mean = values.mean(axis=0)
                row = evaluator.evaluate_explanation(exp_mean, t_exp, len(X_red))
                row.update(evaluator.evaluate_compression(X_red))
                row.update({
                    "method": method,
                    "g": None,
                    "num_bins": None,
                    "compression_time": t_comp,
                    "kernel": None,
                    "explainer": explainer_name,
                    "variant": variant,
                })
                results.append(row)

            elif method == "influence":
                pre = Preprocessor(method="influence", model=model, target_size=5, seed=seed+i) 
                X_red, y_red, idx, t_comp, influence_matrix = pre.run(X, y)
                explainer = Explainer(model=model, name=explainer_name, variant=variant, seed=seed+i)
                values, t_exp = explainer.explain(X_red, y_red)
                exp_mean = values
                if exp_mean.ndim > 1:
                    exp_mean = values.mean(axis=0)
                row = evaluator.evaluate_explanation(exp_mean, t_exp, len(X_red))
                row.update(evaluator.evaluate_compression(X_red))
                row.update({
                    "method": method,
                    "g": None,
                    "num_bins": None,
                    "compression_time": t_comp,
                    "kernel": None,
                    "explainer": explainer_name,
                    "variant": variant,
                })
                results.append(row)

            elif method == "arfpy":
                pre = Preprocessor(method="arfpy", target_size=5, seed=seed+i)
                X_red, y_red, idx, t_comp = pre.run(X, y)
                explainer = Explainer(model=model, name=explainer_name, variant=variant, seed=seed+i)
                values, t_exp = explainer.explain(X_red, y_red)
                exp_mean = values
                if exp_mean.ndim > 1:
                    exp_mean = values.mean(axis=0)
                row = evaluator.evaluate_explanation(exp_mean, t_exp, len(X_red))
                row.update(evaluator.evaluate_compression(X_red))
                row.update({
                    "method": method,
                    "g": None,
                    "num_bins": None,
                    "compression_time": t_comp,
                    "kernel": None,
                    "explainer": explainer_name,
                    "variant": variant,
                })
                results.append(row)
            
            else:
                pre = Preprocessor(method=method, model=model, kernel=kernel, seed=seed+i)
                X_red, y_red, idx, t_comp = pre.run(X, y)
                explainer = Explainer(model=model, name=explainer_name, variant=variant, seed=seed+i)
                values, t_exp = explainer.explain(X_red, y_red)
                exp_mean = values
                if exp_mean.ndim > 1:
                    exp_mean = values.mean(axis=0)
                row = evaluator.evaluate_explanation(exp_mean, t_exp, len(X_red))
                row.update(evaluator.evaluate_compression(X_red))
                row.update({
                    "method": method,
                    "g": None,
                    "num_bins": None,
                    "compression_time": t_comp,
                    "kernel": kernel,
                    "explainer": explainer_name,
                    "variant": variant,
                })
                results.append(row)

    df_results = pd.DataFrame(results)
    save_dir = f"package_metadata/{data_name}/simple_experiment_{kernel}"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"experiment_results_{num_repeats}.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved DataFrame to CSV: {csv_path}")


run_experiment(
    data_name="compas",
    methods= ["iid", "influence", "arfpy", "compress", "compress_with_predictions", "compress_stratified"],
    explainer_name="shap",
    variant="kernel",
    kernel="gaussian",
    num_repeats=13,
)