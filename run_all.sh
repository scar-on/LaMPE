#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-all}"

run_script() {
  local workdir="$1"
  local script="$2"

  echo "==> Running ${workdir}/${script}"
  (
    cd "${ROOT_DIR}/${workdir}"
    bash "${script}"
  )
}

usage() {
  cat <<'EOF'
Usage: bash run_all.sh [target]

Targets:
  ruler          Run RULER evaluation.
  longbench      Run LongBench prediction and evaluation scripts.
  infinitebench  Run InfiniteBench evaluation.
  leval          Run LEval prediction and evaluation scripts.
  all            Run all targets in sequence. This is the default.
EOF
}

case "${TARGET}" in
  ruler)
    run_script "ruler" "run_cmd.sh"
    ;;
  longbench)
    run_script "longbench_eval/LongBench-main" "run_cmd.sh"
    run_script "longbench_eval/LongBench-main" "eval.sh"
    ;;
  infinitebench)
    run_script "InfiniteBench" "run_cmd_llama3.sh"
    ;;
  leval)
    run_script "LEval" "Baselines/run_cmd.sh"
    run_script "LEval/Baselines" "eval_cmd.sh"
    ;;
  all)
    run_script "ruler" "run_cmd.sh"
    run_script "InfiniteBench" "run_cmd_llama3.sh"
    run_script "longbench_eval/LongBench-main" "run_cmd.sh"
    run_script "longbench_eval/LongBench-main" "eval.sh"
    run_script "LEval" "Baselines/run_cmd.sh"
    run_script "LEval/Baselines" "eval_cmd.sh"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown target: ${TARGET}" >&2
    usage >&2
    exit 1
    ;;
esac
