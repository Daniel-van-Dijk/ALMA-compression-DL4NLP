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

# unstructured 0.5

# srun python main.py \
#     --model haoranxu/ALMA-7B \
#     --gradient_path /home/scur1755/GBLM-Pruner-Adapt/gradients/llama2/gradients_aggregrate_norm_l1_model_ALMA-7B.pth \
#     --prune_method gblm \
#     --nsamples 128 \
#     --seed 0 \
#     --sparsity_ratio 0.5 \
#     --sparsity_type unstructured \
#     --save out/alma_7b/unstructured/gblm/

# 2:4 
# srun python main.py \
#     --model haoranxu/ALMA-7B \
#     --gradient_path /home/scur1755/GBLM-Pruner-Adapt/gradients/llama2/gradients_aggregrate_norm_l1_model_ALMA-7B.pth \
#     --prune_method gblm \
#     --nsamples 128 \
#     --seed 0 \
#     --sparsity_ratio 0.5 \
#     --sparsity_type 2:4 \
#     --save out/alma_7b/2to4/gblm/

# 4:8
srun python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --gradient_path /home/scur1755/GBLM-Pruner-Adapt/gradients/llama2/gradients_aggregrate_norm_l1_model_ALMA-7B.pth \
    --prune_method gblm \
    --nsamples 128 \
    --seed 0 \
    --sparsity_ratio 0.5 \
    --sparsity_type 4:8 \
    --save out/alma_7b/4to8/gblm/
