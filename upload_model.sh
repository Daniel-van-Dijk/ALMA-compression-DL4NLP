#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=pruning_llama
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:10:00
#SBATCH --output=slurm/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source activate alma

srun python upload_model_to_huggingface.py \
    --repo_id Max-Bosch/ALMA-7B-pruned-GBLM-2to4 \
    --local_model_folder /home/scur1756/ALMA-compression-DL4NLP/pruned_models/2to4/gblm/