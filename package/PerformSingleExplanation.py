import sage
import shap
import numpy as np
import time
from goodpoints import compress
import os
import numpy as np


class PerformSingleExplanation:
    def __init__(self, dataset_name, method, X_test, y_test, model, estimator, explanation, kernel, g, num_bins, seed_compression, seed_explanation, compressed_size):
        self.dataset_name = dataset_name
        self.method = method
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.estimator = estimator
        self.explanation = explanation
        self.kernel = kernel
        self.g = g
        self.num_bins = num_bins
        self.seed_compression = seed_compression
        self.seed_explanation = seed_explanation
        self.compressed_size = compressed_size
        self.original_size = X_test.shape[0]

    def _compress_iid(self):
        rng = np.random.default_rng(seed=self.seed_compression) 
        self.id_compressed = rng.choice(self.X_test.shape[0], size=self.compressed_size, replace=False)
        return
    
    def _compress_cte(self):
        sigma = np.sqrt(2 * self.X_test.shape[1])
        self.id_compressed = compress.compress_kt(self.X_test, kernel_type=self.kernel, k_params=np.array([sigma**2]), g=self.g, seed=self.seed_compression, num_bins=self.num_bins)
        return
    
    def _compress_cte_predictions(self):
        predictions = self.model.predict(self.X_test)
        X_test_pred = np.concatenate((self.X_test, predictions), axis=1)
        sigma = np.sqrt(2 * X_test_pred.shape[1])
        self.id_compressed = compress.compress_kt(X_test_pred, kernel_type=self.kernel, k_params=np.array([sigma**2]), g=self.g, seed=self.seed_compression, num_bins=self.num_bins)
        return
    
    def _compress_cte_stratified(self):
        self.predicted_labels = np.argmax(self.model.predict(self.X_test), axis=1)
        compressed_indices = []
        for cls in np.unique(self.predicted_labels):
            class_mask = (self.predicted_labels == cls)
            X_class = self.X_test[class_mask]
            id_compressed = compress.compress_kt(X_class, kernel_type=self.kernel, k_params=np.array([np.sqrt(2 * X_class.shape[1])**2]), g=self.g, seed=self.seed_compression, num_bins=self.num_bins)
            full_ids = np.where(class_mask)[0][id_compressed]
            compressed_indices.extend(full_ids)
        self.id_compressed = np.array(compressed_indices)
        return

    def compress_data(self):
        if self.method == "iid":
            start_time = time.time()
            self._compress_iid()
            self.compression_time = time.time() - start_time
        elif self.method == "cte":
            start_time = time.time()
            self._compress_cte()
            self.compression_time = time.time() - start_time
        elif self.method == "cte_predictions":
            start_time = time.time()
            self._compress_cte_predictions()
            self.compression_time = time.time() - start_time
        elif self.method == "cte_stratified":
            start_time = time.time()
            self._compress_cte_stratified()
            self.compression_time = time.time() - start_time
        elif self.method == "ground_truth":
            self.compression_time = 0
            self.id_compressed = np.arange(self.X_test.shape[0])
        else:
            raise ValueError(f"Unknown method: {self.method}")
        self.compressed_size = len(self.id_compressed)
        
    def perform_explanation(self):
        self.compress_data()
        self.X_compressed = self.X_test[self.id_compressed]
        self.y_compressed = self.y_test[self.id_compressed]
        
        if self.explanation == "shap" and self.estimator == "kernel":
            start_time = time.time()    
            explainer = shap.KernelExplainer(lambda x: self.model.predict_proba(x)[:, 1], self.X_compressed, seed=self.seed_explanation)
            self.explanation_values = explainer(self.X_compressed, silent=True).values
            self.explanation_time = time.time() - start_time
        elif self.explanation == "sage" and self.estimator == "permutation":
            start_time = time.time()
            imputer = sage.MarginalImputer(self.model.predict_proba, self.X_compressed)
            explainer = sage.PermutationEstimator(imputer, loss="cross entropy", random_state=self.seed_explanation, n_jobs=16)
            self.explanation_values =  explainer(self.X_compressed, self.y_compressed, bar=False, verbose=False).values
            self.explanation_time = time.time() - start_time
        else:
            raise ValueError(f"Unknown combination of explanation and estimator: {self.explanation}, {self.estimator}")
        
        return {
            "dataset_name": self.dataset_name,
            "method": self.method,
            "estimator": self.estimator,
            "explanation": self.explanation,
            "kernel": self.kernel,
            "g": self.g,
            "num_bins": self.num_bins,
            "seed_compression": self.seed_compression,
            "seed_explanation": self.seed_explanation,
            "compressed_size": self.compressed_size,

            "id_compressed": self.id_compressed,
            "original_size": self.original_size,
            "compression_time": self.compression_time,

            "explanation_values": self.explanation_values,
            "explanation_time": self.explanation_time
        }

        
