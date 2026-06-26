#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

tasks=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)
methods=("dca" )
declare -A envs
envs=(
    ["ori"]="${LAMPE_CONDA_ENV:-lampe}"
    ["dca"]="${LAMPE_DCA_CONDA_ENV:-lampe-dca}"
    ["yarn"]="${LAMPE_CONDA_ENV:-lampe}"
    ["longlm"]="${LAMPE_CONDA_ENV:-lampe}"
    ["ntk"]="${LAMPE_CONDA_ENV:-lampe}"
    ["map"]="${LAMPE_CONDA_ENV:-lampe}"
)

declare -A extra_args
extra_args=(
    ["ori"]=""
    ["dca"]=""
    ["yarn"]=""
    ["longlm"]="--group 21 --window 2048"
    ["map"]="--head 1024 --use_length 3072 --tail 200"
)

data_dir="data-jsonl"
script="test_ruler_llama.py"


source "${CONDA_PROFILE:-${HOME}/miniconda3/etc/profile.d/conda.sh}"

max_lens=( $((128 * 1024)))

for MAX_len in "${max_lens[@]}"; do
    for method in "${methods[@]}"; do
        conda activate "${envs[$method]}"
        
        for task in "${tasks[@]}"; do
            echo "Running task: $task with method=$method and MAX_len=$MAX_len on GPU 6"
            CUDA_VISIBLE_DEVICES=4 python "$script" \
                --task "$task" \
                --data_dir "$data_dir/$task/Meta-Llama-3-8B-Instruct-${MAX_len}.jsonl" \
                --method "$method" \
                --model_path "${LLAMA_MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}" \
                --max_length "$MAX_len" \
                ${extra_args[$method]}
        done
    done
done