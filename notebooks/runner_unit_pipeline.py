# ============================================================
#  SLURM JOB SUBMISSION SCRIPT — SMALL DATASETS (CC18 + CTR23)
#  SHAP & SAGE EXPLAINERS ON ANN AND XGBOOST MODELS
# ============================================================
import os
from bonXAI.core.utils import CC18_SMALL, CTR23_SMALL

model_names = ["ann", "xgboost"]

compressors = [
    ("shap", "kernel", 8, "kernel_thinning", "stratified", 1, "gaussian"),
]

print("[START] Generating and submitting SLURM jobs for small datasets \n")

for model_name in model_names:
    for dataset_id in CC18_SMALL + CTR23_SMALL:
        dataset_name = f"{dataset_id}"
        for explainer_name, strategy, n_jobs, compression_name, data_modification_method, m, kernel in compressors:
            
            if explainer_name == "sage":
                mem_gb = "100G"
            elif explainer_name == "shap":
                mem_gb = "100G"

            job_name = f"{dataset_name}_{explainer_name}_{model_name}"
            sh_file = f"run_{job_name}.sh"

            path_to_script = "/mnt/evafs/faculty/home/kbokhan/bsc-compress-then-explain/notebooks/unit_pipeline.py"
            path_to_python = "/mnt/evafs/faculty/home/kbokhan/bsc_cte/bin/python"

            with open(sh_file, "w") as f:

                f.write(f"""#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p long
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --mem={mem_gb}
#SBATCH --nodelist=dgx-1,dgx-2,dgx-3,dgx-4
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_log.txt
#SBATCH --error=logs/{job_name}_err.txt
#SBATCH --cpus-per-task={n_jobs}

# Run the Python script with all arguments
{path_to_python} {path_to_script} \\
  --dataset_id {dataset_id} \\
  --model_name {model_name} \\
  --compression_name {compression_name} \\
  --data_modification_method {data_modification_method} \\
  --m {m} \\
  --kernel {kernel} \\
  --explainer_name {explainer_name} \\
  --strategy {strategy} \\
  --n_jobs {n_jobs}
""")

        os.system(f"chmod +x {sh_file}")
        os.system(f"sbatch {sh_file}")
        print(f"--> Submitted job: {job_name}")

# ============================================================
#  SLURM JOB SUBMISSION SCRIPT — LARGE DATASETS (CC18 + CTR23)
#  EXPECTED GRADIENTS ON ANN MODELS
# ============================================================
