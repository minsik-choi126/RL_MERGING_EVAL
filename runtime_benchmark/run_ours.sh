#!/usr/bin/env bash
# Time our method end-to-end:
#   Phase A: W extraction with DP=N GPUs  (--dp <N>, default 2)
#   Phase B: ktcol_polar_renorm merge on 1 GPU (cuda:0, fair vs baselines)
#
# Models: pass HF id or local path via env vars. Defaults match canonical
# 7B Qwen2.5 RL experts.
#
# Usage:
#   bash run_ours.sh                          # DP=2
#   bash run_ours.sh --dp 4                   # DP=4
#   BASE_MODEL=Qwen/Qwen2.5-7B-Instruct bash run_ours.sh
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
MERGE_OUT="${THIS_DIR}/merged_ours"
LOG_DIR="${THIS_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "════════════════════════════════════════════════════════════════"
echo "  Ours runtime benchmark   $(date '+%F %T')"
echo "  Phase A (W extract):  DP=${DP}"
echo "  Phase B (merge):      1 GPU (cuda:0, fair vs baselines)"
echo "  base   : ${BASE_MODEL}"
echo "  coding : ${EXPERT_CODING}"
echo "  tool   : ${EXPERT_TOOL}"
echo "  memory : ${EXPERT_MEMORY}"
echo "════════════════════════════════════════════════════════════════"

rm -f "${W_OUT}"
rm -rf "${MERGE_OUT}"

# ─── Phase A ───
echo ""
echo "════ Phase A: W_col extraction (DP=${DP}) ════"
TA0=$(date +%s.%N)
python "${THIS_DIR}/extract_w.py" \
    --base_model "${BASE_MODEL}" \
    --expert_coding "${EXPERT_CODING}" \
    --expert_tool "${EXPERT_TOOL}" \
    --expert_memory "${EXPERT_MEMORY}" \
    --out_npz "${W_OUT}" \
    --dp "${DP}" \
    --key_top_frac 0.10 --select_mode bottom \
    > "${LOG_DIR}/phaseA_W.log" 2>&1
TA1=$(date +%s.%N)
T_W=$(awk "BEGIN{printf \"%.2f\", ${TA1} - ${TA0}}")
T_W_INT=$(printf "%.0f" "${T_W}")
echo "[Phase A] wall = ${T_W}s ($((T_W_INT/60))m$((T_W_INT%60))s)"
[ -s "${W_OUT}" ] || { echo "[FATAL] W npz missing"; exit 1; }

# ─── Phase B ───
echo ""
echo "════ Phase B: merge (1 GPU, cuda:0) ════"
TB0=$(date +%s.%N)
python "${THIS_DIR}/merge_ours.py" \
    --base_model "${BASE_MODEL}" \
    --expert_coding "${EXPERT_CODING}" \
    --expert_tool "${EXPERT_TOOL}" \
    --expert_memory "${EXPERT_MEMORY}" \
    --w_col_file "${W_OUT}" \
    --out_dir "${MERGE_OUT}" \
    --device cuda:0 \
    > "${LOG_DIR}/phaseB_merge.log" 2>&1
TB1=$(date +%s.%N)
T_M=$(awk "BEGIN{printf \"%.2f\", ${TB1} - ${TB0}}")
T_M_INT=$(printf "%.0f" "${T_M}")
echo "[Phase B] wall = ${T_M}s ($((T_M_INT/60))m$((T_M_INT%60))s)"

T_TOTAL=$(awk "BEGIN{printf \"%.2f\", ${T_W} + ${T_M}}")
T_TOTAL_INT=$(printf "%.0f" "${T_TOTAL}")

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Ours runtime SUMMARY   (DP=${DP})"
echo "════════════════════════════════════════════════════════════════"
printf "  %-30s  %12s  %s\n" "phase" "wall(s)" "wall(mm:ss)"
printf "  %-30s  %12s  %02d:%02d\n" "Phase A (W extract, DP=${DP})" "${T_W}"  $((T_W_INT/60))     $((T_W_INT%60))
printf "  %-30s  %12s  %02d:%02d\n" "Phase B (merge, 1 GPU)"        "${T_M}"  $((T_M_INT/60))     $((T_M_INT%60))
printf "  %-30s  %12s  %02d:%02d\n" "TOTAL"                         "${T_TOTAL}" $((T_TOTAL_INT/60)) $((T_TOTAL_INT%60))
echo ""
echo "  W npz:  ${W_OUT}"
echo "  Merged: ${MERGE_OUT}"
echo "  Logs:   ${LOG_DIR}/phaseA_W.log, phaseB_merge.log"
echo "════════════════════════════════════════════════════════════════"
