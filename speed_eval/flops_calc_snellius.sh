#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=test_inference
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:10:00
#SBATCH --output=flops/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source activate alma
which python
srun python flops_calculation.py