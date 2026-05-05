#!/usr/bin/env bash
# Time the TSV merge baseline on a single GPU (cuda:0).
set -euo pipefail
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EXPERT_CODING="${EXPERT_CODING:-Gen-Verse/ReasonFlux-Coder-7B}"
EXPERT_TOOL="${EXPERT_TOOL:-emrgnt-cmplxty/Qwen2.5-7B-Instruct-ToolRL-grpo-cold}"
EXPERT_MEMORY="${EXPERT_MEMORY:-BytedTsinghua-SIA/RL-MemoryAgent-7B}"

OUT_DIR="${THIS_DIR}/merged_tsv"
LOG_DIR="${THIS_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "════════════════════════════════════════════════════════════════"
echo "  TSV runtime benchmark   $(date '+%F %T')"
echo "  Merge: 1 GPU (cuda:0)"
echo "  base   : ${BASE_MODEL}"
echo "  coding : ${EXPERT_CODING}"
echo "  tool   : ${EXPERT_TOOL}"
echo "  memory : ${EXPERT_MEMORY}"
echo "════════════════════════════════════════════════════════════════"

rm -rf "${OUT_DIR}"

T0=$(date +%s.%N)
python "${THIS_DIR}/merge_tsv.py" \
    --base_model "${BASE_MODEL}" \
    --expert_coding "${EXPERT_CODING}" \
    --expert_tool "${EXPERT_TOOL}" \
    --expert_memory "${EXPERT_MEMORY}" \
    --out_dir "${OUT_DIR}" \
    --device cuda:0 \
    > "${LOG_DIR}/tsv.log" 2>&1
T1=$(date +%s.%N)
DUR=$(awk "BEGIN{printf \"%.2f\", ${T1} - ${T0}}")
DUR_INT=$(printf "%.0f" "${DUR}")

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  TSV runtime SUMMARY"
echo "════════════════════════════════════════════════════════════════"
printf "  %-30s  %12s  %02d:%02d\n" "TSV merge (1 GPU)" "${DUR}" $((DUR_INT/60)) $((DUR_INT%60))
echo ""
echo "  Out:    ${OUT_DIR}"
echo "  Log:    ${LOG_DIR}/tsv.log"
echo "════════════════════════════════════════════════════════════════"
