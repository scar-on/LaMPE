#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${LEVAL_DIR}"

# Matches the output directory produced by Baselines/run_cmd.sh:
# Predictions/exam_eval/llama3_8b_cache1/
path="llama3_8b_cache1"
tasks=(coursera gsm100 quality tpo sci_fi)

for task in "${tasks[@]}"; do
  python Evaluation/auto_eval_difflen.py \
    --pred_file "Predictions/exam_eval/${path}/${task}.pred.jsonl"
done
