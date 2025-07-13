from PerformSingleExplanation import PerformSingleExplanation
import numpy as np
import os

class GroundTruthExperiment:
    def __init__(self, dataset_name, X_test, y_test, model, n_repeats=3):
        self.dataset_name = dataset_name
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.n_repeats = n_repeats

    def perform(self):
        all_results = []

        for i in range(self.n_repeats):
            print(f"Running ground truth calculation iteration {i+1}/{self.n_repeats}")
            for explanation, esimator in [("shap", "kernel"), ("sage", "permutation")]:
                print(f"Calculating ground truth for estimator: {esimator}, explainer: {explanation}")
                single_explanation = PerformSingleExplanation(
                    dataset_name=self.dataset_name,
                    method="ground_truth",
                    X_test=self.X_test,
                    y_test=self.y_test,
                    model=self.model,
                    estimator=esimator,
                    explanation=explanation,
                    kernel=None, 
                    g=None,
                    num_bins=None,
                    seed_compression=None,
                    seed_explanation=i+1, 
                    compressed_size=self.X_test.shape[0]
                )

                result = single_explanation.perform_explanation()
                all_results.append(result)

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

        dir = f"metadata/{self.dataset_name}/ground_truth_0_{self.n_repeats}.npz"
        os.makedirs(os.path.dirname(dir), exist_ok=True)
        np.savez_compressed(dir, **aggregated)
        print(f"Saved aggregated results to {dir} for {self.dataset_name} with {self.n_repeats} repeats.")



    

                

