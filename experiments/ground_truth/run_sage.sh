#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p long
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=48
#SBATCH --nodes=1
#SBATCH --mem=1000G
#SBATCH --nodelist=dgx-1,dgx-2,dgx-3,dgx-4
#SBATCH --job-name=gmsc
#SBATCH --output=logs/sage_gmsc.txt
#SBATCH --error=logs/sage_gmsc.txt

mkdir -p logs

/mnt/evafs/faculty/home/kbokhan/bsc_cte/bin/python gt_sage.py
