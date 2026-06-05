#!/usr/bin/env bash
set -euo pipefail

# LongBench prediction entry. Keep this script on the llama_cache1-based
# implementation in pred.py.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python pred.py \
  --model Meta-Llama-3-8B-Instruct \
  --max_length 32768 \
  --use_map \
  --method map \
  --use_length 1536 \
  --head 512 \
  --tail 200 \
  --exp_name llama_cache1_map
