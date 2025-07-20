from PerformSingleExplanation import PerformSingleExplanation
from MetricsCalculator import MetricsCalculator
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

class ParameterGExperiment:
    def __init__(self, experiment_name, description, dataset_name, X_test, y_test, model, n_repeats=3, g_values = [0, 1, 2], kernel="gaussian", num_bins=4, ground_truth_path=None):
        self.dataset_name = dataset_name
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.n_repeats = n_repeats
        self.g_values = g_values
        self.kernel = kernel
        self.num_bins = num_bins

        self.experiment_name = experiment_name
        self.description = description

        if ground_truth_path is None:
            self.ground_truth_path = f"metadata/{self.dataset_name}/ground_truth_0_3.npz"
        else:
            self.ground_truth_path = ground_truth_path

    def perform(self):
        all_results = []
        iid_sizes = []

        for i in range(self.n_repeats):
            for g in self.g_values:
                for explanation, esimator in [("shap", "kernel"), ("sage", "permutation")]:
                    for method in ['cte', 'cte_predictions', 'cte_stratified']:
                        print(f"Running experiment for g={g}, method={method}, estimator={esimator}, explanation={explanation}, repeat={i+1}/{self.n_repeats}")
                        single_explanation = PerformSingleExplanation(
                            dataset_name=self.dataset_name,
                            method=method,
                            X_test=self.X_test,
                            y_test=self.y_test,
                            model=self.model,
                            estimator=esimator,
                            explanation=explanation,
                            kernel=self.kernel, 
                            g=g,
                            num_bins=self.num_bins,
                            seed_compression=i+100,
                            seed_explanation=i+1, 
                            compressed_size=None
                        )

                        result = single_explanation.perform_explanation()
                        all_results.append(result)

                        iid_sizes.append(single_explanation.compressed_size)

        iid_sizes = set(iid_sizes)
        
        for i in range(self.n_repeats):
            for comressed_size in iid_sizes:
                for explanation, estimator in [("shap", "kernel"), ("sage", "permutation")]:
                    print(f"Running iid experiment for compressed size: {comressed_size}. Repeat {i+1}/{self.n_repeats}, explanation={explanation}, estimator={estimator}")
                    iid_explanation = PerformSingleExplanation(
                        dataset_name=self.dataset_name,
                        method='iid',
                        X_test=self.X_test,
                        y_test=self.y_test,
                        model=self.model,
                        estimator=estimator,
                        explanation=explanation,
                        kernel=None,
                        g=None,
                        num_bins=None,
                        seed_compression=i+100,
                        seed_explanation=i+1, 
                        compressed_size=comressed_size
                    )
                    iid_result = iid_explanation.perform_explanation()
                    all_results.append(iid_result)

        keys = all_results[0].keys()
        aggregated = {k: [] for k in keys}
        for res in all_results:
            for k in keys:
                aggregated[k].append(res[k])

        for k in aggregated:
            try:
                aggregated[k] = np.array(aggregated[k], dtype=object)
            except Exception:
                pass

        dir = f"metadata/{self.dataset_name}/{self.experiment_name}/changing_g.npz"
        os.makedirs(os.path.dirname(dir), exist_ok=True)
        np.savez_compressed(dir, **aggregated)
        print(f"Saved aggregated results to {dir} for {self.dataset_name} with {self.n_repeats} repeats.")

        # save description to txt
        description_file = f"metadata/{self.dataset_name}/{self.experiment_name}/description.txt"
        os.makedirs(os.path.dirname(description_file), exist_ok=True)
        with open(description_file, 'w') as f:
            f.write(self.description)

        # calculate metrics and save results
        metrics_calculator = MetricsCalculator(ground_truth_path=self.ground_truth_path,
                                               experiment_path=f"metadata/{self.dataset_name}/{self.experiment_name}/changing_g.npz",)
        
        metrics_calculator.calculate_metrics_save_results()

        data = np.load(f"metadata/{self.dataset_name}/{self.experiment_name}/changing_g_with_metrics.npz", allow_pickle=True)

        filtered_dict = {
            key: data[key] for key in data.files if key != 'explanation_values' and key != 'id_compressed'
        }

        df = pd.DataFrame(filtered_dict)
        df_shap = df[df['explanation'] == 'shap']
        df_sage = df[df['explanation'] == 'sage']

        self.save_chart(df_shap, "mae", "shap")
        self.save_chart(df_sage, "mae", "sage")

    def save_chart(self, df, y_name, exp_name):
        grouped = df.groupby(["method", "compressed_size"]).agg(
            mean=("mae", "mean"),
            std=("mae", "std")
        ).reset_index()

        colors = {
            "iid": "#578ade",
            "cte": "#c6554c",
            "cte_stratified": "#67AE6E",
            "cte_predictions": "#F5C45E",
        }

        method_labels = {
            "iid": "iid",
            "cte": "cte",
            "cte_stratified": "cte-stratified",
            "cte_predictions": "cte-predictions",
        }

        fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=300)

        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

        for method in grouped['method'].unique():
            method_data = grouped[grouped['method'] == method]
            x = method_data['compressed_size']
            y = method_data['mean']
            yerr = method_data['std']
            
            ax.errorbar(
                x=x, y=y, yerr=yerr,
                fmt='o-', label=method_labels.get(method, method),
                color=colors.get(method, None),
                markersize=7, capsize=6, linewidth=2, capthick=2
            )

        ax.set_xlabel("Compressed Size")
        ax.set_ylabel(y_name)
        ax.set_title(f"{y_name} vs Compressed Size ({exp_name.capitalize()})")
        ax.legend(title="Method")
        ax.grid(axis='y', linestyle='dotted', alpha=0.7)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"metadata/{self.dataset_name}/{self.experiment_name}/changing_g_{exp_name}_{y_name}.png", dpi=300)
        plt.close(fig)