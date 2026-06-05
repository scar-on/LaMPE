#!/usr/bin/env bash
set -euo pipefail

max_lens=( $((40 * 1024)) $((48 * 1024)) $((56 * 1024)) $((64 * 1024)))

model_path="${LLAMA_MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
script="auto_prepare_data.py"
temp="llama3"

for MAX_len in "${max_lens[@]}"; do
    echo "Running auto_prepare_data.py with MAX_len=$MAX_len"
    CUDA_VISIBLE_DEVICES=4 python "$script" \
        --model_path "$model_path" \
        --max_length "$MAX_len" \
        --temp "$temp"
done
