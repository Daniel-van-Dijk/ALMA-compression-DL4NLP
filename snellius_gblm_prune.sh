#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=pruning_llama
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=slurm/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

srun python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --gradient_path gradients/llama2/gradients_aggregrate_norm_l1_model_Llama-2-7b-hf.pth \
    --prune_method gblm \
    --nsamples 128 \
    --seed 0 \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_7b/unstructured/gblm/
