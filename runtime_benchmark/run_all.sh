#!/usr/bin/env bash
# One-shot: Ours (Phase A + Phase B) → TSV → printed summary table.
# Usage:
#   bash run_all.sh                # default DP=2
#   bash run_all.sh --dp 4
set -euo pipefail
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DP=2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dp) DP="$2"; shift 2;;
        *) echo "unknown arg: $1"; exit 1;;
    esac
done

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EXPERT_CODING="${EXPERT_CODING:-Gen-Verse/ReasonFlux-Coder-7B}"
EXPERT_TOOL="${EXPERT_TOOL:-emrgnt-cmplxty/Qwen2.5-7B-Instruct-ToolRL-grpo-cold}"
EXPERT_MEMORY="${EXPERT_MEMORY:-BytedTsinghua-SIA/RL-MemoryAgent-7B}"

W_OUT="${THIS_DIR}/W_col_neg_top10.npz"
OURS_OUT="${THIS_DIR}/merged_ours"
TSV_OUT="${THIS_DIR}/merged_tsv"
LOG_DIR="${THIS_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "════════════════════════════════════════════════════════════════"
echo "  Runtime benchmark: Ours → TSV    $(date '+%F %T')"
echo "  DP for Ours.PhaseA = ${DP}    Merge for both = 1 GPU (cuda:0)"
echo "════════════════════════════════════════════════════════════════"

# ─── Ours.PhaseA ─────────────────────────────────────────────────────
rm -f "${W_OUT}"; rm -rf "${OURS_OUT}"
echo ""
echo "──── Ours / Phase A: W_col extraction (DP=${DP}) ────"
TA0=$(date +%s.%N)
python "${THIS_DIR}/extract_w.py" \
    --base_model "${BASE_MODEL}" \
    --expert_coding "${EXPERT_CODING}" \
    --expert_tool "${EXPERT_TOOL}" \
    --expert_memory "${EXPERT_MEMORY}" \
    --out_npz "${W_OUT}" --dp "${DP}" \
    --key_top_frac 0.10 --select_mode bottom \
    > "${LOG_DIR}/phaseA_W.log" 2>&1
TA1=$(date +%s.%N)
T_W=$(awk "BEGIN{printf \"%.2f\", ${TA1} - ${TA0}}")
echo "  done: ${T_W}s"
[ -s "${W_OUT}" ] || { echo "[FATAL] W npz missing"; exit 1; }

# ─── Ours.PhaseB ─────────────────────────────────────────────────────
echo ""
echo "──── Ours / Phase B: ktcol_polar_renorm merge (1 GPU) ────"
TB0=$(date +%s.%N)
python "${THIS_DIR}/merge_ours.py" \
    --base_model "${BASE_MODEL}" \
    --expert_coding "${EXPERT_CODING}" \
    --expert_tool "${EXPERT_TOOL}" \
    --expert_memory "${EXPERT_MEMORY}" \
    --w_col_file "${W_OUT}" \
    --out_dir "${OURS_OUT}" \
    --device cuda:0 \
    > "${LOG_DIR}/phaseB_merge.log" 2>&1
TB1=$(date +%s.%N)
T_M=$(awk "BEGIN{printf \"%.2f\", ${TB1} - ${TB0}}")
echo "  done: ${T_M}s"

# ─── TSV ─────────────────────────────────────────────────────────────
rm -rf "${TSV_OUT}"
echo ""
echo "──── TSV: merge (1 GPU) ────"
TT0=$(date +%s.%N)
python "${THIS_DIR}/merge_tsv.py" \
    --base_model "${BASE_MODEL}" \
    --expert_coding "${EXPERT_CODING}" \
    --expert_tool "${EXPERT_TOOL}" \
    --expert_memory "${EXPERT_MEMORY}" \
    --out_dir "${TSV_OUT}" --device cuda:0 \
    > "${LOG_DIR}/tsv.log" 2>&1
TT1=$(date +%s.%N)
T_T=$(awk "BEGIN{printf \"%.2f\", ${TT1} - ${TT0}}")
echo "  done: ${T_T}s"

# ─── Summary ─────────────────────────────────────────────────────────
T_OURS=$(awk "BEGIN{printf \"%.2f\", ${T_W} + ${T_M}}")
fmt() {
    local s_int=$(printf "%.0f" "$1")
    printf "%dm%02ds" $((s_int/60)) $((s_int%60))
}

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  RUNTIME SUMMARY   $(date '+%F %T')"
echo "════════════════════════════════════════════════════════════════"
printf "  %-32s  %12s  %s\n" "step" "wall(s)" "wall(mm:ss)"
printf "  %-32s  %12s  %s\n" "Ours / Phase A (W extract DP=${DP})" "${T_W}" "$(fmt ${T_W})"
printf "  %-32s  %12s  %s\n" "Ours / Phase B (merge 1 GPU)"        "${T_M}" "$(fmt ${T_M})"
printf "  %-32s  %12s  %s\n" "Ours / Total"                        "${T_OURS}" "$(fmt ${T_OURS})"
printf "  %-32s  %12s  %s\n" "TSV   / Merge (1 GPU)"               "${T_T}" "$(fmt ${T_T})"
echo ""
echo "  outputs: ${OURS_OUT} , ${TSV_OUT}"
echo "  logs:    ${LOG_DIR}/"
echo "════════════════════════════════════════════════════════════════"
