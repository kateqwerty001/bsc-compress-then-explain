dataset_names = ['compas', 'gaussian', 'heloc', 'german', 'heart', 'pima', 'adult']

from openxai.model import LoadModel
from openxai.dataloader import ReturnLoaders
from GroundTruthExperiment import GroundTruthExperiment
import sys
from ParameterGExperiment import ParameterGExperiment

sys.stdout.reconfigure(line_buffering=True)


for d_name in dataset_names:
    print("#############################################################")
    print(f"Running changing g experiment for dataset: {d_name}")
    _, loader_test = ReturnLoaders(data_name=d_name, download=False, batch_size=128)
    X_test = loader_test.dataset.data
    y_test = loader_test.dataset.targets.to_numpy()
    model = LoadModel(data_name=d_name, ml_model="ann", pretrained=True)
    model.eval()

    changing_g = ParameterGExperiment(
        dataset_name=d_name,
        X_test=X_test,
        y_test=y_test,
        model=model,
        n_repeats=10,
        g_values = None,
        kernel = b"gaussian",
        num_bins = 4, 
        experiment_name="changing_g_num_bins_4_n_repeats_10",
        description="Testing different g values for German dataset, using num_bins=4 and n_repeats=10",
    )

    changing_g.perform()