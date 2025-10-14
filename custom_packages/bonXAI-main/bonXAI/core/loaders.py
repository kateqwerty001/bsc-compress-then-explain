import numpy as np
import pandas as pd
import openml
from openxai.model import LoadModel, ReturnLoaders
from sklearn.model_selection import train_test_split
from bonXAI.core.tabular_preprocessor import TabularPreprocessor
from bonXAI.core.utils import CC18, CTR23
from bonXAI.core.pytorch_ann import PyTorchANN

def load_openxai_data_and_model(dataset_name: str, model_name:str) -> tuple[np.ndarray, np.ndarray, object]:
    """
    Loads test data and pretrained model from OpenXAI → (X_test, y_test, model).
    """
    if model_name != "ann":
        raise ValueError("Only 'ann' model is supported in this function.")
    
    _, loader_test = ReturnLoaders(data_name=dataset_name, download=True, batch_size=128)
    X_test = loader_test.dataset.data
    y_test = loader_test.dataset.targets.to_numpy()

    model = LoadModel(data_name=dataset_name, ml_model=model_name, pretrained=True)
    model.eval()
    return X_test, y_test, model


def load_openml_data_and_model(dataset_id: int, model_name:str, task_type:str=None) -> tuple[np.ndarray, np.ndarray, object]:
    """
    Loads test data and pretrained model from OpenML → (X_test, y_test, model).
    """
    if dataset_id in CC18:
        task_type = "classification"
    elif dataset_id in CTR23:
        task_type = "regression"
    elif task_type is None:
        raise ValueError(f"Dataset ID {dataset_id} not found in CC18 or CTR23. Please specify task_type ('classification' or 'regression').")

    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=dataset_id)

    prep = TabularPreprocessor(
        task_type=task_type,
        id_like_threshold=0.99, 
        scale_all_numeric_after_encoding=True, 
        scale_target_in_regression=True, 
        random_state=0
    )
    X_train, y_train = prep.fit_transform(X_train, y_train)
    X_test, y_test = prep.transform(X_test, y_test)

    if model_name == "ann":
        model = PyTorchANN(task_type=task_type, random_state=0)
        model.fit(X_train.values, y_train)
    else:
        raise ValueError("Only 'ann' model is supported in this function.")

    return X_train.values, y_train, X_test.values, y_test, model, prep
