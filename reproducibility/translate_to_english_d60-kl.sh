#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=translate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:50:00
#SBATCH --output=slurm/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source activate ALMA_compression

model_hf="Max-Bosch/ALMA-7B-pruned-GBLM-unstructured-0.6"
model="ALMA-7B-pruned-GBLM-60percent-kl-r16"

declare -A languages=(["de"]="german"
                    ["cs"]="czech"
                    ["is"]="icelandic"
                    ["zh"]="chinese"
                    ["ru"]="russian")
tgt="en"

for src in "${!languages[@]}"; do
    echo "$model: Translating from $src to $tgt"
    echo "data/to_english_sorted/${languages[$src]}/src.txt"
    echo "output/to_english_sorted/${languages[$src]}/out.txt"

    python translate_distilled.py \
        --fin "data/to_english_sorted/${languages[$src]}/src.txt" \
        --fout "output/to_english_sorted/${languages[$src]}/$model.txt" \
        --ckpt "../distillation/60percent-kl-r16/checkpoint-7338" \
        --src "$src" \
        --tgt "$tgt" \
        --dtype bfloat16 \
        --model "$model_hf" \
        --beam 5
done