#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Matches the output directory produced by run_cmd.sh:
# pred/Meta-Llama-3-8B-Instruct_llama_cache1_map/
python eval.py --model Meta-Llama-3-8B-Instruct_llama_cache1_map
