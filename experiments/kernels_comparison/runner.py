import os

datasets = ["german", "gaussian", "heart"]
explainers = [
    ("shap", "kernel", 50),
    ("sage", "permutation", 16)
]

for dataset in datasets:
    for explainer_name, strategy, n_jobs in explainers:
        job_name = f"{dataset}_{explainer_name}"
        sh_file = f"run_{job_name}.sh"

        with open(sh_file, "w") as f:
            f.write(f"""#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p long
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --nodelist=dgx-1,dgx-2,dgx-3,dgx-4
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_log.txt
#SBATCH --error=logs/{job_name}_err.txt
#SBATCH --cpus-per-task={n_jobs}

/mnt/evafs/faculty/home/kbokhan/bsc_cte/bin/python /mnt/evafs/faculty/home/kbokhan/bsc-compress-then-explain/experiments/kernels_comparison/script.py \\
  --dataset {dataset} \\
  --explainer_name {explainer_name} \\
  --strategy {strategy} \\
  --n_jobs {n_jobs}
""")

        os.system(f"chmod +x {sh_file}")
        os.system(f"sbatch {sh_file}")
        print(f"Submitted {job_name}")
