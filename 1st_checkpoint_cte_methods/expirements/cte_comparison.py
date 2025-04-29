import time
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics.pairwise import rbf_kernel

from goodpoints import compress
from openxai.model import LoadModel
from openxai.dataloader import ReturnLoaders
import sage
import shap

class FeatureImportanceExperimentInClass:
    def __init__(self, model_name, datasets, explanations, methods, verbose=True, save_dir=None):
        self.model_name = model_name
        self.datasets = datasets
        self.explanations = explanations
        self.methods = methods 
        self.verbose = verbose
        self.save_dir = save_dir

    def exp_sage_permutation(self, model, X, y, X_background=None, estimator="permutation", random_state=None, bar=False, verbose=False):
        if X_background is None:
            X_background = X
        imputer = sage.MarginalImputer(model.predict_proba, X_background)
        explainer = (sage.PermutationEstimator)(imputer, loss="cross entropy", random_state=random_state)
        return explainer(X, y, bar=bar, verbose=verbose).values

    def exp_shap_kernel(self, model, X, X_background=None, random_state=None, bar=False):
        if X_background is None:
            X_background = X
        explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], X_background, seed=random_state)
        return explainer(X, silent=not bar).values

    def load_data_model(self, data_name):
        _, loader_test = ReturnLoaders(data_name=data_name, download=False, batch_size=128)
        X_test = loader_test.dataset.data
        y_test = loader_test.dataset.targets.to_numpy()
        model = LoadModel(data_name=data_name, ml_model=self.model_name, pretrained=True)
        model.eval()
        self.predictions = model.predict(X_test)
        return X_test, y_test, model

    def compress_data(self, X_test, y_test, model, method, repeat, iid_size=None):
        start_time = time.time()
        sigma = np.sqrt(2 * X_test.shape[1])
        compressed_indices = []
        if method == "cte":
            compressed_indices = compress.compresspp_kt(X_test, kernel_type=b"gaussian", k_params=np.array([sigma**2]), g=4, seed=repeat)
        elif method == "iid":
            if iid_size is None:
                num_rows = X_test.shape[0]
                x = int(np.floor(np.log2(np.sqrt(num_rows))))
                n_samples = 2**x
            else:
                n_samples = iid_size
            rng = np.random.default_rng(seed=repeat) 
            compressed_indices = rng.choice(X_test.shape[0], n_samples, replace=False)
        elif method == "cte_stratified_pred":
            y_pred = np.argmax(self.predictions, axis=1)
            for cls in np.unique(y_pred):
                class_mask = (y_pred == cls)
                X_class = X_test[class_mask]
                id_compressed = compress.compresspp_kt(X_class, kernel_type=b"gaussian", k_params=np.array([sigma**2]), g=4, seed=repeat)
                full_ids = np.where(class_mask)[0][id_compressed]
                compressed_indices.extend(full_ids)
            compressed_indices = np.array(compressed_indices)
        elif method == "cte_stratified_real":
            unique_classes = np.unique(y_test)
            for cls in unique_classes:
                X_class = X_test[y_test == cls]
                id_compressed = compress.compresspp_kt(X_class, kernel_type=b"gaussian", k_params=np.array([sigma**2]), g=4, seed=repeat)
                compressed_indices.extend(np.where(y_test == cls)[0][id_compressed])
            compressed_indices = np.array(compressed_indices)
        elif method == "cte_with_predictions":
            X_test_pred = np.concatenate((X_test, self.predictions), axis=1)
            compressed_indices = compress.compresspp_kt(X_test_pred, kernel_type=b"gaussian", k_params=np.array([np.sqrt(2 * X_test_pred.shape[1])**2]), g=4, seed=repeat)
        end_time = time.time() - start_time
        return X_test[compressed_indices], y_test[compressed_indices], end_time, compressed_indices

    def compute_explanations(self, model, X_test, y_test, X_compressed, explanation, repeat):
        start_time = time.time()
        if explanation == "shap":
            exp = self.exp_shap_kernel(model, X_test, X_compressed, bar=False, random_state=repeat)
        else:
            exp = self.exp_sage_permutation(model, X_test, y_test, X_compressed, bar=False, random_state=repeat)
        end_time_explanation = time.time() - start_time
        return exp, end_time_explanation
    
    def save_results(self, data_name, results, times, start, stop):
        if self.save_dir is not None:
            save_dir = os.path.join(self.save_dir, data_name)
        else:
            save_dir = f'metadata/{data_name}'
        os.makedirs(save_dir, exist_ok=True)
        np.save(f'{save_dir}/cte_shap_sage_results_{start}_{stop}.npy', results)
        times.to_csv(f'{save_dir}/cte_shap_sage_times_{start}_{stop}.csv', index=False)

    def calculate_mmd(self, X, Y, gamma):
        XX = rbf_kernel(X, X, gamma)
        YY = rbf_kernel(Y, Y, gamma)
        XY = rbf_kernel(X, Y, gamma)
        return XX.mean() + YY.mean() - 2 * XY.mean()

    def calculate_top_k(self, exp, gt, k=5):
        if exp.ndim == 2:  
            exp = np.mean(exp, axis=0)
        if gt.ndim == 2:
            gt = np.mean(gt, axis=0)
        
        exp_top_k = np.argsort(exp)[-k:]  
        gt_top_k = np.argsort(gt)[-k:]  
        correct_top_k = len(set(exp_top_k).intersection(set(gt_top_k)))
        return correct_top_k / k

    def run_experiment(self, start=0, stop=3, shap_values_gt=None, ground_truth_repeat=1, iid_size=None):
        for data_name in self.datasets:
            results = {f'{method}_{explanation}': [] for method in self.methods 
                       for explanation in self.explanations}
            if shap_values_gt is not None:
                times = pd.DataFrame(columns=['dataset', 'repeat', 'method', 'time_explanation', 'time_sample_choose', 'n_original', 'n_sample', 'mmd', 'mae', 'top_k'])
            else:
                times = pd.DataFrame(columns=['dataset', 'repeat', 'method', 'time_explanation', 'time_sample_choose', 'n_original', 'n_sample', 'mmd'])
            
            if self.verbose:
                print(f'==== dataset: {data_name} ====')
            X_test, y_test, model = self.load_data_model(data_name)
            
            gamma_val = 1 / (np.sqrt(2 * X_test.shape[1])**2)

            for repeat in tqdm(range(start, stop)):
                if self.verbose:
                    print(f'= repeat: {repeat}')
                for method in self.methods:
                    X_compressed, y_compressed, end_time_sample_choose, id_compressed = self.compress_data(X_test, y_test, model, method, repeat, iid_size=iid_size)
                    
                    for explanation in self.explanations:
                        exp, end_time_explanation = self.compute_explanations(model, X_test, y_test, X_compressed, explanation, repeat)
                        
                        mmd = self.calculate_mmd(X_test, X_compressed, gamma=gamma_val)

                        mae = None
                        top_k_score = None
                        gt_key = None
                        if shap_values_gt is not None:
                            if explanation == "shap":
                                gt_key = 'ground_truth_shap_kernel'
                            elif explanation == "sage":
                                gt_key = 'ground_truth_sage_permutation'

                            if gt_key is not None and gt_key in shap_values_gt:
                                shap_values_gt_ = np.mean(np.array([shap_values_gt[gt_key][i] for i in range(ground_truth_repeat)]), axis=0)
                                mae = np.mean(np.abs(exp - shap_values_gt_))
                                top_k_score = self.calculate_top_k(exp, shap_values_gt_, k=5)
                    
                        row = {
                            'dataset': data_name,
                            'repeat': repeat,
                            'method': f'{method}_{explanation}', 
                            'time_explanation': end_time_explanation,
                            'time_sample_choose': end_time_sample_choose, 
                            'n_original': X_test.shape[0],
                            'n_sample': len(id_compressed),
                            'mmd': mmd
                        }
                        if shap_values_gt is not None:
                            row['mae'] = mae
                            row['top_k'] = top_k_score

                        times = pd.concat([times, pd.DataFrame([row])])
                        results[f'{method}_{explanation}'].append(exp)
            
            self.save_results(data_name, results, times, start, stop)

if __name__ == "__main__":
    shap_values_gt = np.load(f'./metadata/german/ground_truth_shap_sage_results_0_1.npy', allow_pickle=True).item()
    datasets = ['german']
    explanations = ["shap", "sage"]
    methods = ["cte", "iid", "cte_stratified_real", "cte_stratified_pred", "cte_with_predictions"]
    model_name = 'ann'

    runner = FeatureImportanceExperimentInClass(
        model_name=model_name,
        datasets=datasets,
        explanations=explanations,
        methods=methods,
        verbose=True
    )
    runner.run_experiment(start=0, stop=3, shap_values_gt=shap_values_gt, ground_truth_repeat=1) 

    shap_values_gt_c = np.load(f'./metadata/compas/ground_truth_shap_sage_results_0_3.npy', allow_pickle=True).item()
    datasets_c = ['compas']

    runner_c = FeatureImportanceExperimentInClass(
        model_name=model_name,
        datasets=datasets_c,
        explanations=explanations,
        methods=methods,
        verbose=True
    )
    runner_c.run_experiment(start=0, stop=3, shap_values_gt=shap_values_gt_c, ground_truth_repeat=3) 
