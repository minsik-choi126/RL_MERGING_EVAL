#!/usr/bin/env bash
# KT-Merge end-to-end pipeline for Qwen3-1.7B + 3 RL experts (ifeval/math/coding).
#
#   Step 0: download HF prompts → generate targets → data/training/{ifeval,math,coding}.jsonl
#   Step 1: prep proxy per_query npz from local training data
#   Step 2: compute per-row W (position-level Δ>θ events)
#   Step 3: merge ours (positionkey) + 12 baselines
#
# Evaluation (ifeval, aime24/25/26, livebench, livecodebench) is handled by
# the user's existing eval pipeline — NOT this script. After Step 3 finishes,
# point the user's eval pipeline at outputs/merges/<method>/.
#
# Each step is idempotent: re-run skips already-built artifacts.
#
# ── GPU toggle ──────────────────────────────────────────────────────────────
#   CUDA_VISIBLE_DEVICES=0       (default — pick any single GPU)
#   CUDA_VISIBLE_DEVICES=0,1     expose two GPUs
#   DEVICE=cuda:0                logical device passed to scripts
#
# Internally: prep_proxy, compute_W, and merge scripts use DEVICE.
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
#   THRESHOLD=0.1                 position-level Δ key threshold (see guide below)
#   EPS_SCALE=0.01                W ε-add scale
#   ENERGY=0.90                   KT-Truncation energy threshold
#   PER_EXPERT=1                  1=per-expert W (2D, N×d_out); 0=union W (1D)
#
# ── THRESHOLD GUIDE for Qwen3-1.7B ──────────────────────────────────────────
# Δ distribution depends on the base + experts; tune so each expert's own-data
# key-position rate falls in the 10–20% band.
#
#   1. Run with default THRESHOLD=0.1 once. Step 2 prints per-expert rates:
#        [per-expert mask] ifeval  (ifeval): NNNN/NNNNN (XX.X%)
#        [per-expert mask] math    (math)  : NNNN/NNNNN (XX.X%)
#        [per-expert mask] coding  (coding): NNNN/NNNNN (XX.X%)
#   2. Adjust:
#        rates >25%  → raise THRESHOLD (try 0.2, 0.3, 0.5)
#        rates <5%   → lower THRESHOLD (try 0.05, 0.02)
#   3. Re-run Step 2 only with the new threshold:
#        SKIP_DOWNLOAD=1 SKIP_PREP=1 SKIP_MERGE=1 \
#        THRESHOLD=<new> bash run_pipeline.sh
#   4. Once each expert lands in 10–20%, re-merge using the chosen W file.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="${HERE}/scripts"
TRAINING_DIR="${HERE}/data/training"
PER_QUERY_DIR="${HERE}/data/per_query"
OUTPUTS_DIR="${HERE}/outputs"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
N_QUERIES="${N_QUERIES:-128}"
SEED="${SEED:-42}"
THRESHOLD="${THRESHOLD:-0.1}"
EPS_SCALE="${EPS_SCALE:-0.01}"
ENERGY="${ENERGY:-0.90}"
DEVICE="${DEVICE:-cuda:0}"
PER_EXPERT="${PER_EXPERT:-1}"      # 1 = per-expert W (2D, N×d_out); 0 = union W (1D, d_out)
_W_SUFFIX="$([ "${PER_EXPERT}" = "1" ] && echo "_perexpert" || echo "")"
W_FILE="${OUTPUTS_DIR}/W_activation_positionkey_${THRESHOLD}${_W_SUFFIX}.npz"

mkdir -p "${TRAINING_DIR}" "${PER_QUERY_DIR}" "${OUTPUTS_DIR}"

echo "════════════════════════════════════════════════════════════════"
echo "  KT-Merge pipeline   (started: $(date))"
echo "════════════════════════════════════════════════════════════════"
echo "  BASE_MODEL          : ${BASE_MODEL}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-(unset; use any)}"
echo "  DEVICE              : ${DEVICE}"
echo "  N_QUERIES / SEED    : ${N_QUERIES} / ${SEED}"
echo "  THRESHOLD / EPS / E : ${THRESHOLD} / ${EPS_SCALE} / ${ENERGY}"
echo "  PER_EXPERT          : ${PER_EXPERT}"
echo "  W_FILE              : ${W_FILE}"
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

# ── Step 2: compute W ───────────────────────────────────────────────────────
if [ "${SKIP_COMPUTE_W:-0}" != "1" ]; then
    echo ""
    echo "── Step 2: compute W (Δ>${THRESHOLD}, per_expert=${PER_EXPERT}) ──"
    PE_FLAG=""
    [ "${PER_EXPERT}" = "1" ] && PE_FLAG="--per_expert"
    python "${SCRIPTS}/compute_W_activation_positionkey.py" \
        --threshold "${THRESHOLD}" \
        --eps_scale "${EPS_SCALE}" \
        --device "${DEVICE}" \
        --in_dir "${PER_QUERY_DIR}" \
        --out "${W_FILE}" \
        ${PE_FLAG}
    echo ""
    echo "  >>> THRESHOLD GUIDE (Qwen3-1.7B):"
    echo "      Look at the 'key positions' percentages printed above per expert."
    echo "      Pick THRESHOLD so that each expert's own-data key rate is ~10-20%."
    echo "      If too high (>30%) → raise THRESHOLD (try 0.2, 0.3, 0.5)."
    echo "      If too low (<5%)   → lower  THRESHOLD (try 0.05, 0.02)."
    echo "      Re-run Step 2 with: SKIP_DOWNLOAD=1 SKIP_PREP=1 SKIP_MERGE=1 \\"
    echo "                          THRESHOLD=<new> bash run_pipeline.sh"
else
    echo ""
    echo "── Step 2: SKIPPED (SKIP_COMPUTE_W=1)"
fi

# ── Step 3: merge ───────────────────────────────────────────────────────────
if [ "${SKIP_MERGE:-0}" != "1" ]; then
    echo ""
    echo "── Step 3: merge ours + baselines ──"
    BASE_MODEL="${BASE_MODEL}" \
    W_FILE="${W_FILE}" \
    THRESHOLD="${THRESHOLD}" \
    PER_EXPERT="${PER_EXPERT}" \
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
