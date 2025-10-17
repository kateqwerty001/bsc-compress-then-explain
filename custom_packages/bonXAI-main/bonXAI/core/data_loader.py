import numpy as np
import openml
import pandas as pd
import torch
from bonXAI.core.pytorch_ann import PyTorchANN
from bonXAI.core.tabular_preprocessor import TabularPreprocessor
from bonXAI.core.utils import CC18_ALL, CTR23_ALL
from openxai.model import LoadModel, ReturnLoaders
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor


class DataLoader:
    """
    A class to handle loading of preprocessed data and trained models from various data sources like OpenML and OpenXAI.
    """

    def __init__(self):
        pass

    def load_from_openxai(
        self, dataset_name: str, model_name: str
    ) -> tuple[np.ndarray, np.ndarray, torch.nn.Module]:
        """
        Loads test data and a pretrained model from the OpenXAI library.

        Args:
            dataset_name: The name of the dataset to load (e.g., "german").
            model_name: The name of the model. Only "ann" is supported.

        Returns:
            A tuple containing:
            - X_test: Test features as a NumPy array.
            - y_test: Test targets as a NumPy array.
            - model: The pretrained PyTorch model, set to evaluation mode.
        """
        if model_name != "ann":
            raise ValueError(f"Model '{model_name}' is not supported. Only 'ann' is available.")

        _, loader_test = ReturnLoaders(data_name=dataset_name, download=True, batch_size=128)
        X_test = loader_test.dataset.data.to_numpy()
        y_test = loader_test.dataset.targets.to_numpy()

        model = LoadModel(data_name=dataset_name, ml_model=model_name, pretrained=True)
        model.eval()

        return X_test, y_test, model

    def _get_model(
        self, model_name: str, task_type: str, random_state: int
    ) -> PyTorchANN | XGBClassifier | XGBRegressor:
        """
        Initializes a model based on its name and the machine learning task type.
        """
        model_map = {
            "ann": PyTorchANN(task_type=task_type, random_state=0),
            "xgboost": (
                XGBClassifier(n_estimators=200, random_state=random_state)
                if task_type == "classification"
                else XGBRegressor(n_estimators=200, random_state=random_state)
            ),
        }
        model = model_map.get(model_name)
        if model is None:
            raise ValueError(
                f"Model '{model_name}' is not supported. "
                f"Choose from {list(model_map.keys())}."
            )
        return model

    def load_from_openml(
        self, dataset_id: int, model_name: str, task_type: str = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, TabularPreprocessor]:
        """
        Loads data from OpenML, preprocesses it, trains a model, and returns all components.

        Args:
            dataset_id: The ID of the dataset on OpenML.
            model_name: The name of the model to train ('ann' or 'xgboost').
            task_type: The task type ('classification' or 'regression').
                       If None, the class will attempt to infer it based on the dataset ID.

        Returns:
            A tuple containing:
            - X_train: Preprocessed training features.
            - y_train: Preprocessed training target.
            - X_test: Preprocessed test features.
            - y_test: Preprocessed test target.
            - model: The trained model object.
            - preprocessor: The fitted TabularPreprocessor instance.
        """
        if task_type is None:
            if dataset_id in CC18_ALL:
                task_type = "classification"
            elif dataset_id in CTR23_ALL:
                task_type = "regression"
            else:
                raise ValueError(
                    f"Task type for dataset ID {dataset_id} is unknown. "
                    "Please specify `task_type` ('classification' or 'regression')."
                )

        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=dataset_id
        )

        preprocessor = TabularPreprocessor(
            task_type=task_type,
            id_like_threshold=0.99,
            scale_all_numeric_after_encoding=True,
            scale_target_in_regression=True,
            random_state=0,
        )
        X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
        X_test_processed, y_test_processed = preprocessor.transform(X_test, y_test)

        model = self._get_model(model_name, task_type, random_state=dataset_id)
        model.fit(X_train_processed.values, y_train_processed)

        return (
            X_train_processed.to_numpy(),
            y_train_processed,
            X_test_processed.to_numpy(),
            y_test_processed,
            model,
            preprocessor,
        )