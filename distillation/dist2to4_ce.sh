#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=dist2to4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=slurm/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source activate alma
which python
srun python distillation.py \
    --backbone Max-Bosch/ALMA-7B-pruned-WANDA-2to4 \
    --distillation_weight 1.0 \
    --output_dir 2to4-ce