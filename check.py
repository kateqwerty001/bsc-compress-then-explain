from bonXAI.core.data_loader import DataLoader
from goodpoints import compressc, compress
import numpy as np
import math
from bonXAI.core.preprocessor import Preprocessor
import time

loader = DataLoader()
dataset_name, X_train, y_train, X_test, y_test, model, _ = loader.load_from_openml(
    dataset_id=int(28),
    model_name="ann"
)

model.model_.eval()

kernels = ["gaussian", "matern", "inverse_multiquadric", "sobolev"]
coeffs = [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32]

target_size = int(math.sqrt(compress.largest_power_of_four(len(X_test))))
print("Base target size:", target_size)
print("Full test size:", len(X_test))

for kernel in kernels:
    for coeff in coeffs:
        current_target_size = int(target_size * coeff)
        print(f"\nKernel: {kernel}, Coeff: {coeff}, Target size: {current_target_size}")

        pre = Preprocessor(
            X=X_test.copy(),
            y=y_test.copy(),
            model=model,
            compression_method="kernel_thinning",
            data_modification_method="none",
            seed=0
        )

        try:
            start_time = time.perf_counter()
            X_kt, y_kt, idx_kt, t_kt = pre._preprocess(
                g=4,
                num_bins=32,
                target_size=current_target_size,
                kernel=kernel
            )
            end_time = time.perf_counter()
            print(f"Compression time: {end_time - start_time:.4f} s")

        except Exception as e:
            print(f"Error for kernel {kernel} with coeff {coeff}: {e}")
            continue

        print("Selected indices length:", len(idx_kt))
        print("Uniqueness of selected indices:", len(set(idx_kt)))
