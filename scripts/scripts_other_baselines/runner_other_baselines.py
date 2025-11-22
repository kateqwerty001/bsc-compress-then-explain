# ============================================================
#  SLURM JOB SUBMISSION SCRIPT — SMALL DATASETS (CC18 + CTR23)
#  SHAP & SAGE $ SHAP-IQ EXPLAINERS ON ANN AND XGBOOST MODELS 
#  STEIN THINNING COMPRESSION
# ============================================================
import os
from bonXAI.core.utils import CC18_SMALL, CTR23_SMALL

seed = 42

model_names = ["ann", "xgboost"]

explainers = [
    ("shap", "kernel", 16),
    ("sage", "permutation", 16),
    ("shapiq", "kernel", 16),
]

compression_methods = ["stein_thinning"]
data_modification_method = ["none"]

print("[START] Generating and submitting SLURM jobs for small datasets \n")

for data_modification in data_modification_method:
    for compression_method in compression_methods:
        for model_name in model_names:
            for dataset_id in CC18_SMALL + CTR23_SMALL:
                dataset_name = f"{dataset_id}"
                for explainer_name, strategy, n_jobs in explainers:
                    
                    mem_gb = "50G"

                    job_name = f"st_{dataset_name}_{explainer_name}_{model_name}_{compression_method}_{data_modification}"
                    sh_file = f"run_{job_name}.sh"

                    path_to_script = "/mnt/evafs/faculty/home/kbokhan/bsc-compress-then-explain/experiments/scripts_other_baselines/other_baselines.py"
                    path_to_python = "/mnt/evafs/faculty/home/kbokhan/bsc_cte/bin/python"

                    with open(sh_file, "w") as f:

                        f.write(f"""#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem={mem_gb}
#SBATCH --nodelist=dgx-1,dgx-2,dgx-3,dgx-4
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_log.txt
#SBATCH --error=logs/{job_name}_err.txt
#SBATCH --cpus-per-task={n_jobs}
#SBATCH --gres=gpu:0
#SBATCH --mail-user=kateqwerty001@gmail.com
#SBATCH --mail-type=ALL

# Run the Python script with all arguments
{path_to_python} {path_to_script} \\
--dataset_id {dataset_id} \\
--model_name {model_name} \\
--explainer_name {explainer_name} \\
--strategy {strategy} \\
--compression_method {compression_method} \\
--data_modification_method {data_modification} \\
--n_jobs {n_jobs} \\
--seed {seed} \\
""")

                    os.system(f"chmod +x {sh_file}")
                    os.system(f"sbatch {sh_file}")
                    print(f"--> Submitted job: {job_name}")



