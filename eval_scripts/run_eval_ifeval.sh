#!/usr/bin/env bash
# IFEval evaluation — vLLM inference + official IFEval scoring (no VERL needed).
#
# Usage:
#   MODEL_PATH=/path/to/model bash run_eval_ifeval.sh
#   MODEL_PATH=/path/to/model TP=4 bash run_eval_ifeval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_eval_env.sh"

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH is not set." >&2
    exit 1
fi

MODEL_SHORT="$(basename "${MODEL_PATH%/}")"
TP="${TP:-4}"

REPO_ROOT_GUESS="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT_GUESS}/results/verl_outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${MODEL_SHORT}/evaluation_output/ifeval}"

echo "============================================================"
echo "  IFEval Evaluation"
echo "  Model:  ${MODEL_PATH}"
echo "  TP:     ${TP}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Started: $(date)"
echo "============================================================"

python "${SCRIPT_DIR}/ifeval_eval.py" \
    --model "${MODEL_PATH}" \
    --tp "${TP}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "  IFEval complete! $(date)"
echo "  Results: ${OUTPUT_DIR}/summary.json"
echo "============================================================"
