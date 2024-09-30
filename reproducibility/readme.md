# ALMA Compression DL4NLP - Reproducibility

## Overview
This directory contains materials and instructions to reproduce the results of the ALMA Compression DL4NLP project.

## Structure
- `data/`: Contains test sets of EN->XX and XX->EN used in the experiments.
- `results/`: Contains the results of the experiments.

## Requirements
- Install ALMA_compression conda environment: [env.yml]

## Setup
1. Navigate to the reproducibility directory:
    ```
    cd ALMA-compression-DL4NLP/reproducibility
    ```
2. Install the required packages:
    ```
    conda env create -f env.yml
    ```
3. Translate the different test sets.
    -  modify the `model_hf="haoranxu/ALMA-7B" model="ALMA-7B"` variables in both the [translate_from_english.sh]() and [translate_to_english.sh]() files
    - run `sbatch translate_from_english.sh` and `sbatch translate_to_english.sh`

4. Evaluate the output files.
    - change the value for `model="ALMA-7B"` in [evaluate.sh]() file
    - run `sbatch evaluate.sh`


## Results
The results of the experiments can be found in the `results/` directory under the specified model name.

## Contact
For any questions or issues, please contact Oliver
