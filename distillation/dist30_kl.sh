#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=dist30_kl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=slurm/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source activate alma

echo "dist30kl"
which python
srun python distillation.py \
    --backbone Max-Bosch/ALMA-7B-pruned-GBLM-unstructured-0.3 \
    --distillation_weight 1.0 \
    --output_dir 30percent-kl-r16