#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=model_sizes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=slurm/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source activate ALMA_compression

srun python model_sizes.py