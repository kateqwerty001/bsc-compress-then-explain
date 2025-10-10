#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p long
#SBATCH --cpus-per-task=50
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --nodelist=dgx-1,dgx-2,dgx-3,dgx-4
#SBATCH --job-name=kernels
#SBATCH --output=logs/kernels_log.txt
#SBATCH --error=logs/kernels_err.txt

mkdir -p logs

/mnt/evafs/faculty/home/kbokhan/bsc_cte/bin/python /mnt/evafs/faculty/home/kbokhan/bsc-compress-then-explain/experiments/kernels_comparison/script.py
