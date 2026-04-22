#!/bin/bash
set -euo pipefail

###############################################################################
# BFCL Tool-Use Evaluation (범용)
#
# Usage:
#   bash run_eval.sh /path/to/model [num_gpus]
#
# Examples:
#   bash run_eval.sh Qwen/Qwen3-1.7B                    # Qwen3-1.7B (8 GPU)
#   bash run_eval.sh /path/to/Qwen2.5-7B-model 4        # Qwen2.5-7B (4 GPU)
#   bash run_eval.sh /path/to/merged_model 2             # 2 GPU
#
# GPU 제약 (tensor parallelism):
#   Qwen3-1.7B  (heads=16, kv=8)  → 1, 2, 4, 8
#   Qwen2.5-7B  (heads=28, kv=4)  → 1, 2, 4
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# ─── Args ────────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Usage: bash run_eval.sh <model_path> [num_gpus]"
    echo ""
    echo "  model_path: HuggingFace name or local path to model"
    echo "  num_gpus:   tensor parallel size (default: 4)"
    echo ""
    echo "  Qwen3-1.7B  → num_gpus: 1, 2, 4, 8"
    echo "  Qwen2.5-7B  → num_gpus: 1, 2, 4"
    exit 1
fi

MODEL_PATH="$1"
NUM_GPUS="${2:-4}"

# Set env var for generic handler
export BFCL_MODEL_PATH="${MODEL_PATH}"

echo "============================================================"
echo "  BFCL Tool-Use Evaluation"
echo "  Model:      ${MODEL_PATH}"
echo "  BFCL name:  custom-model"
echo "  Categories: live, non_live"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Started:    $(date)"
echo "============================================================"

# ─── Generate ────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  [generate] ${MODEL_PATH}"
echo "============================================================"

bfcl generate \
    --model custom-model \
    --test-category live,non_live \
    --backend vllm \
    --num-gpus "${NUM_GPUS}" \
    --gpu-memory-utilization 0.9

echo "  -> Generation done"

# ─── Evaluate ────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  [evaluate] ${MODEL_PATH}"
echo "============================================================"

bfcl evaluate \
    --model custom-model \
    --test-category live,non_live \
    || echo "  WARNING: bfcl evaluate exited non-zero (leaderboard CSV 생성 실패일 수 있음, score JSON은 유효)"

echo "  -> Evaluation done"

# ─── Rename results to model name ────────────────────────────────────────────
# custom-model → 실제 모델 이름으로 결과 폴더 rename
MODEL_SHORT="$(basename "${MODEL_PATH}")"
RESULT_BASE="${SCRIPT_DIR}/berkeley-function-call-leaderboard"

for subdir in result score; do
    SRC="${RESULT_BASE}/${subdir}/custom-model"
    DST="${RESULT_BASE}/${subdir}/${MODEL_SHORT}"
    if [[ -d "${SRC}" ]]; then
        rm -rf "${DST}"
        mv "${SRC}" "${DST}"
        echo "  -> ${subdir}/custom-model -> ${subdir}/${MODEL_SHORT}"
    fi
done

echo ""
echo "============================================================"
echo "  Complete! $(date)"
echo "  Results: ${RESULT_BASE}/result/${MODEL_SHORT}/"
echo "  Scores:  ${RESULT_BASE}/score/${MODEL_SHORT}/"
echo "============================================================"

# ─── Print score summary table ──────────────────────────────────────────────
SCORE_DIR="${RESULT_BASE}/score/${MODEL_SHORT}"
if [[ -d "${SCORE_DIR}" ]]; then
    python3 "${SCRIPT_DIR}/print_bfcl_table.py" "${MODEL_SHORT}" --score-dir "${SCORE_DIR}"
fi
