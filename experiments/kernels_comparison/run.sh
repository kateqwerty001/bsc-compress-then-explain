#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p long
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --nodelist=dgx-1,dgx-2,dgx-3,dgx-4
#SBATCH --job-name=sh-gaussian
#SBATCH --output=logs/gaussian_shap_log.txt
#SBATCH --error=logs/gaussian_shap_err.txt

dataset_name="gaussian"
explainer_name="sage"
variant="permutation"

mkdir -p logs

echo "Running ground_truth_calculator.py for dataset: $dataset_name"

/mnt/evafs/faculty/home/kbokhan/bsc_cte/bin/python script.py \
    --data $dataset_name \
    --explainer-name $explainer_name \
    --variant $variant 