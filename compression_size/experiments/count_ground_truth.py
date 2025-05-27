import time
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from openxai.model import LoadModel
from openxai.dataloader import ReturnLoaders
import sage
import shap

datasets = ['compas']
explanations = ["sage", "shap"]
estimators = ["kernel", "permutation"]
methods = ["ground_truth"]
model_name = 'ann'

class GroundTruthExperiment:
    def __init__(self, model_name, datasets, explanations, estimators, methods, verbose=True):
        self.model_name = model_name
        self.datasets = datasets
        self.explanations = explanations
        self.estimators = estimators 
        self.methods = methods 
        self.verbose = verbose

    def exp_sage(self, model, X, y, X_background=None, estimator="permutation", random_state=None, bar=False, verbose=False):
        if X_background is None:
            X_background = X
        imputer = sage.MarginalImputer(model.predict_proba, X_background)
        explainer = (sage.KernelEstimator if estimator == "kernel" else sage.PermutationEstimator)(imputer, loss="cross entropy", random_state=random_state)
        return explainer(X, y, bar=bar, verbose=verbose).values

    def exp_shap(self, model, X, X_background=None, estimator="permutation", random_state=None, bar=False):
        if X_background is None:
            X_background = X
        if estimator == "kernel":
            explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], X_background, seed=random_state)
        else:
            masker = shap.maskers.Independent(X_background, max_samples=X_background.shape[0])
            explainer = shap.PermutationExplainer(lambda x: model.predict_proba(x)[:, 1], masker, seed=random_state)
        return explainer(X, silent=not bar).values

    def load_data_model(self, data_name):
        _, loader_test = ReturnLoaders(data_name=data_name, download=False, batch_size=128)
        X_test, y_test = loader_test.dataset.data, loader_test.dataset.targets.to_numpy()
        model = LoadModel(data_name=data_name, ml_model=self.model_name, pretrained=True)
        model.eval()
        self.predictions = model.predict(X_test)
        return X_test, y_test, model

    def compute_explanations(self, model, X_test, y_test, explanation, estimator, repeat):
        start_time = time.time()
        if explanation == "shap":
            exp = self.exp_shap(model, X_test, X_test, estimator=estimator, bar=False, random_state=repeat)
        else:
            exp = self.exp_sage(model, X_test, y_test, X_test, estimator=estimator, bar=False, random_state=repeat)
        end_time_explanation = time.time() - start_time
        return exp, end_time_explanation

    def save_results(self, data_name, results, times, start, stop):
        save_dir = f'metadata/{data_name}'
        os.makedirs(save_dir, exist_ok=True)
        np.save(f'{save_dir}/ground_truth_shap_sage_results_{start}_{stop}.npy', results)
        times.to_csv(f'{save_dir}/ground_truth_shap_sage_times_{start}_{stop}.csv', index=False)

    def run_experiment(self, start=0, stop=10):
        for data_name in self.datasets:
            results = {f'{method}_{explanation}_{estimator}': [] for method in self.methods 
                       for explanation in self.explanations for estimator in self.estimators}
            times = pd.DataFrame(columns=['dataset', 'repeat', 'method', 'time_explanation', 'n_original'])
            
            if self.verbose:
                print(f'==== dataset: {data_name} ====')
            X_test, y_test, model = self.load_data_model(data_name)
            
            for repeat in tqdm(range(start, stop)):
                if self.verbose:
                    print(f'= repeat: {repeat}')
                for method in self.methods:
                    for explanation in self.explanations:
                        for estimator in self.estimators:
                            exp, end_time_explanation = self.compute_explanations(model, X_test, y_test, explanation, estimator, repeat)
                            
                            times = pd.concat([times, pd.DataFrame({
                                'dataset': [data_name],
                                'repeat': [repeat],
                                'method': [f'{method}_{explanation}_{estimator}'], 
                                'time_explanation': [end_time_explanation],
                                'n_original': [X_test.shape[0]],
                            })])
                            results[f'{method}_{explanation}_{estimator}'].append(exp)
            
            self.save_results(data_name, results, times, start, stop)

experiment = GroundTruthExperiment(model_name, datasets, explanations, estimators, methods)
experiment.run_experiment(start=0, stop=3)
