#!/usr/bin/env bash
# Download the 4 models required (base + 3 RL experts) from HuggingFace.
# Skips any already on disk.
#
# Default HF ids (override via env if your repo has different names):
#   BASE_MODEL      Qwen/Qwen2.5-7B-Instruct
#   EXPERT_CODING   Gen-Verse/ReasonFlux-Coder-7B
#   EXPERT_TOOL     emrgnt-cmplxty/Qwen2.5-7B-Instruct-ToolRL-grpo-cold
#   EXPERT_MEMORY   BytedTsinghua-SIA/RL-MemoryAgent-7B
set -euo pipefail

if ! command -v huggingface-cli >/dev/null; then
    echo "[hint] pip install -U huggingface_hub"
    exit 1
fi

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EXPERT_CODING="${EXPERT_CODING:-Gen-Verse/ReasonFlux-Coder-7B}"
EXPERT_TOOL="${EXPERT_TOOL:-emrgnt-cmplxty/Qwen2.5-7B-Instruct-ToolRL-grpo-cold}"
EXPERT_MEMORY="${EXPERT_MEMORY:-BytedTsinghua-SIA/RL-MemoryAgent-7B}"

CACHE_DIR="${HF_HOME:-${HOME}/.cache/huggingface}"
echo "HF cache: ${CACHE_DIR}"
mkdir -p "${CACHE_DIR}"

for repo in "${BASE_MODEL}" "${EXPERT_CODING}" "${EXPERT_TOOL}" "${EXPERT_MEMORY}"; do
    echo ""
    echo "── ${repo}"
    huggingface-cli download "${repo}" --cache-dir "${CACHE_DIR}/hub" \
        --include "*.safetensors" "config.json" "tokenizer*" "*.json" "vocab.json" "merges.txt"
done

echo ""
echo "[done] models in ${CACHE_DIR}/hub. Pass HF ids to run_*.sh — they will resolve to local cache."
