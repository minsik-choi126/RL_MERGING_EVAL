#!/usr/bin/env bash
# Build 7 baselines (TSV/ours skipped) into /131_data/geeho/minsik/qwen3_new/<method>/.
# Two groups run in parallel: cuda:0 and cuda:1+cpu.
set -uo pipefail

ROOT="/home2/geeho/minsik/RL_MERGING_EVAL/KT_merge"
OUT="/131_data/geeho/minsik/qwen3_new"
LOG="${ROOT}/outputs/merge_logs"
mkdir -p "${LOG}"

BASE="Qwen/Qwen3-1.7B"
EXPERTS=("${ROOT}/models/ifeval" "${ROOT}/models/math" "${ROOT}/models/lucy")

exec > >(tee -a "${OUT}/_baselines_master.log") 2>&1

echo "════════════════════════════════════════════════════════════════"
echo "  baselines parallel — start at $(date)"
echo "════════════════════════════════════════════════════════════════"

run_method() {
    local tag="$1"; shift
    local save="${OUT}/${tag}"
    local log="${LOG}/${tag}.log"
    if ls "${save}"/*.safetensors >/dev/null 2>&1; then
        echo "[skip] ${tag} (already exists)"; return 0
    fi
    mkdir -p "${save}"
    echo "[$(date '+%H:%M:%S')] [start] ${tag}  log=${log}"
    python "${ROOT}/scripts/merge.py" "$@" \
        --base_model "${BASE}" \
        --expert_models "${EXPERTS[@]}" \
        --save_dir "${save}" \
        > "${log}" 2>&1
    rc=$?
    if [ "${rc}" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] [done]  ${tag}"
    else
        echo "[$(date '+%H:%M:%S')] [FAIL] ${tag} rc=${rc}"
    fi
}

# ── Group A: cuda:0 (task_arithmetic, dare_ta, star, ram) ──
{
    run_method task_arithmetic --method task_arithmetic
    run_method dare_ta         --method dare --dare_merge_method task_arithmetic --device cuda:0
    run_method star            --method star --device cuda:0
    run_method ram             --method ram --device cuda:0
} &
A=$!

# ── Group B: cuda:1 (iso_cts, ram_plus) + cpu (ties) ──
{
    run_method ties     --method ties    --device cpu
    run_method iso_cts  --method iso_cts --device cuda:1
    run_method ram_plus --method ram_plus --device cuda:1
} &
B=$!

wait "${A}"; rc_a=$?
wait "${B}"; rc_b=$?

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  baselines parallel — done at $(date)  (group_a=${rc_a}, group_b=${rc_b})"
echo "════════════════════════════════════════════════════════════════"
ls -la "${OUT}/"
