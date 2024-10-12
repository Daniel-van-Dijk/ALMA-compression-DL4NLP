#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=evaluate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=slurm/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source activate ALMA_compression

model="ALMA-7B-pruned-WANDA-2to4-lora"

output_path="results/$model.txt"

declare -A languages=(["de"]="german"
                    ["cs"]="czech"
                    ["is"]="icelandic"
                    ["zh"]="chinese"
                    ["ru"]="russian")

# TRANSLATION FROM ENGLISH TO NON-ENGLISH LANGUAGES

src="en"

for tgt in "${!languages[@]}"; do
    echo "$model: evaluating from $src to $tgt" >> ${output_path}

    src_path="data/from_english_sorted/${languages[$tgt]}/src.txt"
    hyp_path="output/from_english_sorted/${languages[$tgt]}/$model.txt"
    tgt_path="data/from_english_sorted/${languages[$tgt]}/tgt.txt"

    TOK="13a"
    if [ ${tgt} == "zh" ]; then
        TOK="zh"
    fi

    SACREBLEU_FORMAT=text sacrebleu -tok ${TOK} -w 2 ${tgt_path} < ${hyp_path} >> ${output_path}
    comet-score -s $src_path -t $hyp_path -r $tgt_path | tail -n 1 >> ${output_path}

    echo "________________________________________________________" >> ${output_path}
done


# TRANSLATION FROM NON-ENGLISH LANGUAGES TO ENGLISH

tgt="en"

for src in "${!languages[@]}"; do
    echo "$model: evaluating from $src to $tgt" >> ${output_path}

    src_path="data/to_english_sorted/${languages[$src]}/src.txt"
    hyp_path="output/to_english_sorted/${languages[$src]}/$model.txt"
    tgt_path="data/to_english_sorted/${languages[$src]}/tgt.txt"

    TOK="13a"
    if [ ${tgt} == "zh" ]; then
        TOK="zh"
    fi
    
    SACREBLEU_FORMAT=text sacrebleu -tok ${TOK} -w 2 ${tgt_path} < ${hyp_path} >> ${output_path}
    comet-score -s $src_path -t $hyp_path -r $tgt_path | tail -n 1 >> ${output_path}
    
    echo "________________________________________________________" >> ${output_path}
done