#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=dist50_kl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=slurm/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source activate alma

echo "dist50kl"
which python
srun python distillation.py \
    --backbone Max-Bosch/ALMA-7B-pruned-GBLM-unstructured \
    --distillation_weight 1.0 \
    --output_dir 50percent-kl-r16