dataset_names = ['compas', 'gaussian', 'heloc', 'german', 'heart', 'pima', 'adult']

from openxai.model import LoadModel
from openxai.dataloader import ReturnLoaders
from GroundTruthExperiment import GroundTruthExperiment
import sys

sys.stdout.reconfigure(line_buffering=True)

dataset_names = ['adult']

for d_name in dataset_names:
    print("#############################################################")
    print(f"Running ground truth experiment for dataset: {d_name}")
    _, loader_test = ReturnLoaders(data_name=d_name, download=False, batch_size=128)
    X_test = loader_test.dataset.data
    y_test = loader_test.dataset.targets.to_numpy()
    model = LoadModel(data_name=d_name, ml_model="ann", pretrained=True)
    model.eval()

    ground_truth = GroundTruthExperiment(
        dataset_name=d_name,
        X_test=X_test,
        y_test=y_test,
        model=model,
        n_repeats=3, 
    )
    ground_truth.perform()