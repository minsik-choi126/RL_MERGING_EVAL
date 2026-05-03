#!/usr/bin/env bash
# KT-Merge end-to-end pipeline for Qwen3-1.7B + 3 RL experts (ifeval/math/coding).
#
#   Step 0: download HF prompts → generate targets → data/training/{ifeval,math,coding}.jsonl
#   Step 1: prep proxy per_query npz from local training data
#   Step 2: compute W_expert (per-expert top-K% by Δlog p; default 20%)
#   Step 3: merge ours (W_expert) + 12 baselines
#
# Evaluation (ifeval, aime24/25/26, livebench, livecodebench) is handled by
# the user's existing eval pipeline — NOT this script. After Step 3 finishes,
# point the user's eval pipeline at outputs/merges/<method>/.
#
# Each step is idempotent: re-run skips already-built artifacts.
#
# ── GPU toggle ──────────────────────────────────────────────────────────────
#   CUDA_VISIBLE_DEVICES=0       single GPU (default; everything on cuda:0)
#   CUDA_VISIBLE_DEVICES=0,1     two GPUs (faster Step 1; Step 2 only uses one)
#   DEVICE=cuda:0                logical device for prep_proxy + merge + Step 2
#
# ── Step toggles ────────────────────────────────────────────────────────────
#   SKIP_DOWNLOAD=1     skip step 0a (raw jsonl must already exist)
#   SKIP_GEN_TARGETS=1  skip step 0b (completed jsonl must already exist)
#   SKIP_PREP=1         skip step 1  (per_query npz must already exist)
#   SKIP_COMPUTE_W=1    skip step 2  (W file must already exist)
#   SKIP_MERGE=1        skip step 3  (only run prior steps)
#
# ── Hyperparameters ─────────────────────────────────────────────────────────
#   BASE_MODEL=Qwen/Qwen3-1.7B    HF id of base
#   N_QUERIES=128                 proxy queries per task for log-prob extraction
#   SEED=42                       sampling seed
#   KEY_TOP_FRAC=0.05             per-expert BOTTOM-K fraction by Δlog p (default 5%)
#                                 (anti-key positions where expert under-performs base)
#   ENERGY=0.90                   KT-Truncation energy threshold
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="${HERE}/scripts"
TRAINING_DIR="${HERE}/data/training"
PER_QUERY_DIR="${HERE}/data/per_query"
OUTPUTS_DIR="${HERE}/outputs"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
N_QUERIES="${N_QUERIES:-128}"
SEED="${SEED:-42}"
KEY_TOP_FRAC="${KEY_TOP_FRAC:-0.05}"
ENERGY="${ENERGY:-0.90}"
DEVICE="${DEVICE:-cuda:0}"
KEY_TOP_PCT="$(printf '%02d' "$(awk "BEGIN{printf \"%d\", ${KEY_TOP_FRAC}*100+0.5}")")"
W_COL_FILE="${OUTPUTS_DIR}/W_col_neg_top${KEY_TOP_PCT}_perexpert.npz"

mkdir -p "${TRAINING_DIR}" "${PER_QUERY_DIR}" "${OUTPUTS_DIR}"

echo "════════════════════════════════════════════════════════════════"
echo "  KT-Merge pipeline   (started: $(date))"
echo "════════════════════════════════════════════════════════════════"
echo "  BASE_MODEL          : ${BASE_MODEL}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-(unset; use any)}"
echo "  DEVICE              : ${DEVICE}"
echo "  N_QUERIES / SEED    : ${N_QUERIES} / ${SEED}"
echo "  KEY_TOP_FRAC        : ${KEY_TOP_FRAC}  (BOTTOM ${KEY_TOP_PCT}% per expert)"
echo "  ENERGY              : ${ENERGY}"
echo "  W_COL_FILE          : ${W_COL_FILE}"
echo ""
echo "  step toggles: SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-0} SKIP_PREP=${SKIP_PREP:-0}"
echo "                SKIP_COMPUTE_W=${SKIP_COMPUTE_W:-0} SKIP_MERGE=${SKIP_MERGE:-0}"

# ── Step 0a: download prompts (+ gold answer for math) ─────────────────────
if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    echo ""
    echo "── Step 0a: download Qwen3 RL prompts (seed=${SEED}, val split if available) ──"
    python "${SCRIPTS}/download_training_data.py" \
        --n "${N_QUERIES}" \
        --seed "${SEED}" \
        --out_dir "${TRAINING_DIR}"
else
    echo ""
    echo "── Step 0a: SKIPPED (SKIP_DOWNLOAD=1)"
fi

# ── Step 0b: generate missing targets via task-specific experts ─────────────
# ifeval/coding RL data has no gold answer; we generate one with each task's
# own expert (greedy decoding). Math uses gold answer when present.
if [ "${SKIP_GEN_TARGETS:-0}" != "1" ]; then
    echo ""
    echo "── Step 0b: generate missing targets via task-specific experts ──"
    python "${SCRIPTS}/generate_targets.py" \
        --ifeval "${HERE}/models/ifeval" \
        --math   "${HERE}/models/math" \
        --coding "${HERE}/models/coding" \
        --tokenizer_src "${BASE_MODEL}" \
        --device "${DEVICE}" \
        --data_dir "${TRAINING_DIR}" \
        --max_new_tokens "${MAX_NEW_TOKENS:-512}"
else
    echo ""
    echo "── Step 0b: SKIPPED (SKIP_GEN_TARGETS=1)"
fi

# ── Step 1: proxy data ──────────────────────────────────────────────────────
if [ "${SKIP_PREP:-0}" != "1" ]; then
    echo ""
    echo "── Step 1: prep proxy per_query npz ──"
    python "${SCRIPTS}/prep_proxy_qwen3.py" \
        --base "${BASE_MODEL}" \
        --ifeval "${HERE}/models/ifeval" \
        --math   "${HERE}/models/math" \
        --coding "${HERE}/models/coding" \
        --n_queries "${N_QUERIES}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --training_dir "${TRAINING_DIR}" \
        --out_dir "${PER_QUERY_DIR}"
else
    echo ""
    echo "── Step 1: SKIPPED (SKIP_PREP=1)"
fi

# ── Step 2: compute W_col (column-side, BOTTOM-K% by Δlogp, × ‖W[:,c]‖) ─────
if [ "${SKIP_COMPUTE_W:-0}" != "1" ]; then
    echo ""
    echo "── Step 2: compute W_col (bottom ${KEY_TOP_PCT}% per expert, with ‖W[:,c]‖ factor) ──"
    python "${SCRIPTS}/compute_W_col.py" \
        --key_top_frac "${KEY_TOP_FRAC}" \
        --ifeval "${HERE}/models/ifeval" \
        --math   "${HERE}/models/math" \
        --coding "${HERE}/models/coding" \
        --device "${DEVICE}" \
        --in_dir "${PER_QUERY_DIR}" \
        --out "${W_COL_FILE}"
else
    echo ""
    echo "── Step 2: SKIPPED (SKIP_COMPUTE_W=1)"
fi

# ── Step 3: merge ───────────────────────────────────────────────────────────
if [ "${SKIP_MERGE:-0}" != "1" ]; then
    echo ""
    echo "── Step 3: merge ours + baselines ──"
    BASE_MODEL="${BASE_MODEL}" \
    W_COL_FILE="${W_COL_FILE}" \
    KEY_TOP_FRAC="${KEY_TOP_FRAC}" \
    ENERGY="${ENERGY}" \
    DEVICE="${DEVICE}" \
    OUT_DIR="${OUTPUTS_DIR}/merges" \
    LOG_DIR="${OUTPUTS_DIR}/merge_logs" \
        bash "${SCRIPTS}/merge_baselines.sh"
else
    echo ""
    echo "── Step 3: SKIPPED (SKIP_MERGE=1)"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Pipeline done at $(date)"
echo ""
echo "  Next step (external): evaluate models in ${OUTPUTS_DIR}/merges/ on"
echo "  ifeval / aime24 / aime25 / aime26 / livebench / livecodebench"
echo "  using your existing eval pipeline."
echo "════════════════════════════════════════════════════════════════"
