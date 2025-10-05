#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p long
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=48
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --nodelist=dgx-1,dgx-2,dgx-3,dgx-4
#SBATCH --job-name=adult
#SBATCH --output=logs/sage_adult.txt
#SBATCH --error=logs/sage_adult.txt

mkdir -p logs

/mnt/evafs/faculty/home/kbokhan/bsc_cte/bin/python gt_sage.py
