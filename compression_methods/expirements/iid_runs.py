
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

from cte_comparison import FeatureImportanceExperimentInClass

shap_values_gt_german = np.load(f'./metadata/german/ground_truth_shap_sage_results_0_1.npy', allow_pickle=True).item()
shap_values_gt_compas = np.load(f'./metadata/compas/ground_truth_shap_sage_results_0_3.npy', allow_pickle=True).item()
explanations = ["shap", "sage"]
methods = ["iid"]
model_name = 'ann'
german_sizes = [8, 16, 24, 32, 40]
compas_sizes = [24, 32, 64, 96, 128, 160]

for size in german_sizes:
    runner = FeatureImportanceExperimentInClass(
        model_name=model_name,
        datasets=['german'],
        explanations=explanations,
        methods=methods,
        verbose=True,
        save_dir=f"metadata/iid_runs/{size}"
    )
    runner.run_experiment(start=0, stop=3, shap_values_gt=shap_values_gt_german, ground_truth_repeat=1, iid_size=size) 
for size in compas_sizes:
    runner = FeatureImportanceExperimentInClass(
        model_name=model_name,
        datasets=['compas'],
        explanations=explanations,
        methods=methods,
        verbose=True,
        save_dir=f"metadata/iid_runs/{size}"
    )
    runner.run_experiment(start=0, stop=3, shap_values_gt=shap_values_gt_compas, ground_truth_repeat=1, iid_size=size) 