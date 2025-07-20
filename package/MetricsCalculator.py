from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import os


class MetricsCalculator:
    def __init__(self, ground_truth_path, experiment_path):
        self.experiment_path = experiment_path
        self.ground_truth_path = ground_truth_path

        data = np.load(ground_truth_path, allow_pickle=True)
        explanations = data["explanation"]
        estimators = data["estimator"]

        mask = (explanations == 'sage') & (estimators == 'permutation')
        self.sage_values_gt = data["explanation_values"][mask].mean(axis=0)

        mask = (explanations == 'shap') & (estimators == 'kernel')
        self.shap_values_gt = data["explanation_values"][mask].mean(axis=0).mean(axis=0)

        experiment_data = np.load(experiment_path, allow_pickle=True)
        self.experiment_values = experiment_data["explanation_values"]
        self.experiment_estimators = experiment_data["estimator"]
        self.experiment_explanations = experiment_data["explanation"]

    def _calculate_mmd(self, X, Y, gamma):
        XX = rbf_kernel(X, X, gamma)
        YY = rbf_kernel(Y, Y, gamma)
        XY = rbf_kernel(X, Y, gamma)
        return XX.mean() + YY.mean() - 2 * XY.mean()
    
    def _calculate_top_k(self, exp, gt, k=5):
        if exp.ndim == 2:  
            exp = np.mean(exp, axis=0)
        if gt.ndim == 2:
            gt = np.mean(gt, axis=0)
        
        exp_top_k = np.argsort(exp)[-k:]  
        gt_top_k = np.argsort(gt)[-k:]  
        correct_top_k = len(set(exp_top_k).intersection(set(gt_top_k)))
        return correct_top_k / k
    
    def _metric_mae(self, x, y):
        return np.mean(np.abs(x-y))
    
    def _calculate_metrics(self, k):
        results = []

        for i in range(len(self.experiment_values)):
            exp = self.experiment_values[i]
            explainer = self.experiment_explanations[i]
            estimator = self.experiment_estimators[i]

            if explainer == "sage" and estimator == "permutation":
                gt = self.sage_values_gt
                exp = exp
            elif explainer == "shap" and estimator == "kernel":
                gt = self.shap_values_gt
                exp = exp.mean(axis=0)
            else:
                raise ValueError(f"Unsupported explanation-estimator combination: {explainer}, {estimator}")
            
            gamma = 1 / (np.sqrt(2 * len(self.experiment_values))**2)

            result = {
                "mae": self._metric_mae(gt, exp),
                "top_k": self._calculate_top_k(exp, gt, k=k),
                "mmd": self._calculate_mmd(exp.reshape(1, -1), gt.reshape(1, -1), gamma=gamma)
            }

            results.append(result)

        return results
    
    def calculate_metrics_save_results(self, k=5):
        results = self._calculate_metrics(k=k)

        metrics_dict = {
            "mae": [r["mae"] for r in results],
            "top_k": [r["top_k"] for r in results],
            "mmd": [r["mmd"] for r in results],
        }

        experiment_data = np.load(self.experiment_path, allow_pickle=True)
        new_data = {key: experiment_data[key] for key in experiment_data.files}
        new_data.update(metrics_dict)

        base, ext = os.path.splitext(self.experiment_path)
        output_path = base + "_with_metrics" + ext

        np.savez_compressed(output_path, **new_data)
        print(f"Results saved to {output_path}. Metrics were added: MAE, Top-K Accuracy, MMD.")
