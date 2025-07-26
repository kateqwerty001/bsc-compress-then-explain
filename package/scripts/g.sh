#!/bin/bash

#SBATCH -A mi2lab-normal
#SBATCH -p long
#SBATCH --job-name=pima
#SBATCH --output=log_pima.txt
#SBATCH --error=err_pima.txt
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1 --nodelist=dgx-1,dgx-2,dgx-3,dgx-4
#SBATCH --mem=64G

source ~/bsc_cte/bin/activate
python g.py