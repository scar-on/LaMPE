#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${LEVAL_DIR}"

# LEval prediction entry. Keep this on the llama_cache1-based Llama3 baseline.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python Baselines/llama3-8b-chat_hmt_tpo.py \
  --max_length 32k \
  --head 512 \
  --tail 200 \
  --metric exam_eval \
  --task_path LEval-data/Closed-ended-tasks/ \
  --exp_name llama3_8b_cache1
