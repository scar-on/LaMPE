#!/bin/bash

tasks=(
    "longbook_choice_eng"
    "kv_retrieval"
    "longbook_qa_eng"
    "longbook_sum_eng"
    #"math_find"
    "code_debug"
    "passkey"
    "number_string"
)


declare -A envs
envs=(
    ["ori"]="${LAMPE_CONDA_ENV:-lampe}"
    ["dca"]="${LAMPE_DCA_CONDA_ENV:-lampe-dca}"
    ["yarn"]="${LAMPE_CONDA_ENV:-lampe}"
    ["longlm"]="${LAMPE_CONDA_ENV:-lampe}"
    ["map"]="${LAMPE_CONDA_ENV:-lampe}"
    ["ntk"]="${LAMPE_CONDA_ENV:-lampe}"
)


declare -A extra_args
extra_args=(
    ["ori"]=""
    ["dca"]=""
    ["yarn"]=""
    ["longlm"]="--group 3 --window 65536"
    ["map"]="--head 512 --tail 200 --map_len 4501"
)
# 
methods=("map") #

script="src/test_infbench_llama_cw.py"

source "${CONDA_PROFILE:-${HOME}/miniconda3/etc/profile.d/conda.sh}"

for method in "${methods[@]}"; do

    for MAX_len in $((32 * 1024)) $((64 * 1024)); do
        for task in "${tasks[@]}"; do
            echo "Running task: $task with method=$method and MAX_len=$MAX_len on GPU 6"
            CUDA_VISIBLE_DEVICES=4 python "$script" \
                --task "$task" \
                --method "$method" \
                --model_name llama3 \
                --max_len "$MAX_len" \
                --pretraining_length 8192 \
                ${extra_args[$method]}
        done
    done
done
