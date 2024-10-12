#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=dist20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=slurm/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source activate alma

echo "dist20"
which python
srun python distillation.py \
    --backbone Max-Bosch/ALMA-7B-pruned-GBLM-unstructured-0.2 \
    --distillation_weight 0.5 \
    --output_dir 20percent-ce-kl