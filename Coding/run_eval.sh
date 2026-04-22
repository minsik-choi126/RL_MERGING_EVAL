#!/bin/bash
set -euo pipefail

###############################################################################
# CURE Evaluation Runner (LiveBench / LiveCodeBench)
#
# Usage:
#   bash run_eval.sh <model_path> [dataset] [gpu_per_engine] [total_gpus]
#
# Examples:
#   bash run_eval.sh /path/to/model                           # 둘다, 1GPU/engine x4 (4GPU)
#   bash run_eval.sh /path/to/model LiveBench                 # LiveBench만
#   bash run_eval.sh /path/to/model LiveCodeBench             # LiveCodeBench만
#   bash run_eval.sh /path/to/model all 2                     # 2GPU/engine x2 (4GPU)
#   bash run_eval.sh /path/to/model all 4                     # 4GPU/engine x1 (4GPU, 7B)
#
# GPU 레이아웃 예시 (total_gpus=4):
#   gpu_per_engine=1 → [[0],[1],[2],[3]]   4 engines x 1 GPU (default)
#   gpu_per_engine=2 → [[0,1],[2,3]]       2 engines x 2 GPU
#   gpu_per_engine=4 → [[0,1,2,3]]        1 engine  x 4 GPU (7B 모델)
#
# 환경변수로 GPU 수 지정 가능:
#   TOTAL_GPUS=8 bash run_eval.sh ...     # 8GPU 환경
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# ─── Args ────────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Usage: bash run_eval.sh <model_path> [dataset] [gpu_per_engine]"
    echo ""
    echo "  model_path:     HuggingFace name or local path"
    echo "  dataset:        LiveBench, LiveCodeBench, or omit for both"
    echo "  gpu_per_engine: GPUs per vLLM engine (default: 2)"
    exit 1
fi

MODEL_PATH="$1"
DATASET_ARG="${2:-all}"
GPU_PER_ENGINE="${3:-1}"
# $4 = TOTAL_GPUS (unused, kept for positional compat)
TEMP_ARG="${5:-}"
CHAT_TEMPLATE_ARG="${6:-}"

if [[ "${DATASET_ARG}" == "all" ]]; then
    DATASETS=("LiveBench" "LiveCodeBench")
else
    DATASETS=("${DATASET_ARG}")
fi

# Build gpu_groups from gpu_per_engine
# TOTAL_GPUS: 환경변수로 override 가능 (기본: 자동 감지)
if [[ -z "${TOTAL_GPUS:-}" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        TOTAL_GPUS=$(nvidia-smi --list-gpus | wc -l)
    else
        TOTAL_GPUS=4
    fi
fi
NUM_ENGINES=$(( TOTAL_GPUS / GPU_PER_ENGINE ))
GPU_GROUPS="["
for (( e=0; e<NUM_ENGINES; e++ )); do
    [[ $e -gt 0 ]] && GPU_GROUPS+=","
    GPU_GROUPS+="["
    for (( g=0; g<GPU_PER_ENGINE; g++ )); do
        [[ $g -gt 0 ]] && GPU_GROUPS+=","
        GPU_GROUPS+="$(( e * GPU_PER_ENGINE + g ))"
    done
    GPU_GROUPS+="]"
done
GPU_GROUPS+="]"

EVAL_DIR="${SCRIPT_DIR}/evaluation"

# Where LiveBench.json / LiveCodeBench.json live. Honor CODING_DATA_DIR if set,
# otherwise default to <repo>/Coding/data. If the files are missing, abort early
# with a helpful pointer rather than letting eval.py stack-trace.
export CODING_DATA_DIR="${CODING_DATA_DIR:-${SCRIPT_DIR}/data}"
for _ds in "${DATASETS[@]}"; do
    _f="${CODING_DATA_DIR}/${_ds}.json"
    if [[ ! -f "${_f}" ]]; then
        echo "[ERROR] ${_f} not found." >&2
        echo "        Set CODING_DATA_DIR to the directory containing ${_ds}.json," >&2
        echo "        or drop/symlink the file into ${SCRIPT_DIR}/data/. See README §2." >&2
        exit 1
    fi
done

echo "============================================================"
echo "  CURE Coding Evaluation"
echo "  Model:      ${MODEL_PATH}"
echo "  Datasets:   ${DATASETS[*]}"
echo "  Data dir:   ${CODING_DATA_DIR}"
echo "  GPU layout: ${GPU_GROUPS} (${NUM_ENGINES} engines x ${GPU_PER_ENGINE} GPUs)"
echo "  Started:    $(date)"
echo "============================================================"

# ─── Run ─────────────────────────────────────────────────────────────────────
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  [${DATASET}] ${MODEL_PATH}"
    echo "============================================================"

    cd "${EVAL_DIR}"

    TEMP_FLAG=""
    if [[ -n "${TEMP_ARG}" ]]; then
        TEMP_FLAG="--temp ${TEMP_ARG}"
    fi

    CHAT_TEMPLATE_FLAG=""
    if [[ -n "${CHAT_TEMPLATE_ARG}" ]]; then
        CHAT_TEMPLATE_FLAG="--chat_template ${CHAT_TEMPLATE_ARG}"
    fi

    python eval.py \
        --use_api False \
        --pretrained_model "${MODEL_PATH}" \
        --single_eval True \
        --dataset "${DATASET}" \
        --gpu_groups "${GPU_GROUPS}" \
        ${TEMP_FLAG} \
        ${CHAT_TEMPLATE_FLAG}

    echo "  -> ${DATASET} done for ${MODEL_PATH}"
done

echo ""
echo "============================================================"
echo "  All evaluations complete! $(date)"
echo "  Results:   ${EVAL_DIR}/results/"
echo "  Full data: ${EVAL_DIR}/temp_data/"
echo "============================================================"
