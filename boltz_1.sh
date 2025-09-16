#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=100G
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=preempt
#SBATCH --job-name=boltz2_amide_screen

echo "Starting job at $(date)"
echo "Running on node: $(hostname)"

# Activate your environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate boltz2

# Run your script
python boltz_predict.py 1 --amide 'NS(=O)(C1=C(Cl)C=CC(C(NC2=NC=C(N2)C(O)=O)=O)=C1)=O'

echo "Job finished at $(date)"