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
from GroundTruthExperiment import GroundTruthExperiment

d_name = "adult"

_, loader_test = ReturnLoaders(data_name=d_name, download=False, batch_size=128)
X_test = loader_test.dataset.data
y_test = loader_test.dataset.targets.to_numpy()
model = LoadModel(data_name=d_name, ml_model="ann", pretrained=True)
model.eval()

ground_truth = GroundTruthExperiment(
    dataset_name=d_name,
    X_test=X_test,
    y_test=y_test,
    model=model,n_repeats=3, 
)
ground_truth.perform()