#!/bin/bash
set -euo pipefail

###############################################################################
# run_tool_use_4x.sh — run the BFCL tool_use benchmark for many models,
# optionally repeating each model multiple times (for variance estimation).
#
# Results:
#   $SAVE_DIR/<model_name>/run_<N>/{result,score}/
#   $SAVE_DIR/summary.json      # averaged scores per category
#
# Usage:
#   bash run_tool_use_4x.sh --models-file models.txt
#   N_RUNS=4 bash run_tool_use_4x.sh --models-file models.txt
#   bash run_tool_use_4x.sh --models-file models.txt --dry-run
#
# models.txt format (one model per line; blank + #-comment lines ignored):
#   # name=path               prompt_mode=plain|tool   (prompt_mode is optional)
#   Qwen2.5-7B-Instruct=/path/to/Qwen2.5-7B-Instruct  prompt_mode=plain
#   my-merge=/path/to/my/merge
#
# Environment:
#   SAVE_DIR        Where per-run results are aggregated. Default: $PWD/results/tool_use_4x
#   N_RUNS          How many times each model is re-run.                  Default: 1
#   TP              Tensor parallel size passed into Tool_use/run_eval.sh. Default: auto
#   BFCL_PROMPT_MODE  Default prompt mode when not set in models.txt.     Default: tool
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_USE_DIR="${SCRIPT_DIR}/Tool_use"
BFCL_BASE="${TOOL_USE_DIR}/berkeley-function-call-leaderboard"
SAVE_DIR="${SAVE_DIR:-${SCRIPT_DIR}/results/tool_use_4x}"
N_RUNS="${N_RUNS:-1}"

# Auto-detect TP: divide GPUs across 1 engine. User can override.
if [[ -z "${TP:-}" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        TP="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
        TP="${TP:-1}"
    else
        TP=1
    fi
fi

MODELS_FILE=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models-file) MODELS_FILE="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=true;     shift ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# \?//' | head -40
            exit 0
            ;;
        *) echo "[ERROR] unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${MODELS_FILE}" || ! -f "${MODELS_FILE}" ]]; then
    echo "[ERROR] --models-file <path> is required (example: models.txt.example)" >&2
    exit 1
fi

# Parse models.txt into parallel arrays.
MODEL_ORDER=()
declare -A MODELS
declare -A PROMPT_MODES
while IFS= read -r line || [[ -n "$line" ]]; do
    # Strip comments & whitespace.
    line="${line%%#*}"
    line="${line## }"; line="${line%% }"
    [[ -z "$line" ]] && continue

    # Expected: name=path [prompt_mode=tool|plain]
    name="${line%%=*}"
    rest="${line#*=}"
    path="${rest%% *}"
    pm="${BFCL_PROMPT_MODE:-tool}"
    if [[ "$rest" == *"prompt_mode="* ]]; then
        pm="${rest##*prompt_mode=}"
        pm="${pm%% *}"
    fi
    MODELS["$name"]="$path"
    PROMPT_MODES["$name"]="$pm"
    MODEL_ORDER+=("$name")
done < "${MODELS_FILE}"

if [[ ${#MODEL_ORDER[@]} -eq 0 ]]; then
    echo "[ERROR] no models parsed from ${MODELS_FILE}" >&2; exit 1
fi

mkdir -p "${SAVE_DIR}"

TOTAL=${#MODEL_ORDER[@]}
CURRENT=0
FAILED=()

echo "════════════════════════════════════════════════════════════════"
echo "  Tool Use ${N_RUNS}x Evaluation"
echo "  Models: ${TOTAL} | Repeats: ${N_RUNS} | Total runs: $((TOTAL * N_RUNS))"
echo "  TP: ${TP} | Save dir: ${SAVE_DIR}"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════════"

for MODEL_NAME in "${MODEL_ORDER[@]}"; do
    MODEL_PATH="${MODELS[$MODEL_NAME]}"
    CURRENT=$((CURRENT + 1))

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${CURRENT}/${TOTAL}] ${MODEL_NAME}"
    echo "  path: ${MODEL_PATH}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ ! -d "${MODEL_PATH}" ]]; then
        echo "  [ERROR] not a directory → skip"
        FAILED+=("${MODEL_NAME} (no such dir)")
        continue
    fi

    MODEL_SAVE_DIR="${SAVE_DIR}/${MODEL_NAME}"
    mkdir -p "${MODEL_SAVE_DIR}"

    for RUN in $(seq 1 ${N_RUNS}); do
        RUN_SAVE="${MODEL_SAVE_DIR}/run_${RUN}"

        # Already-complete runs are skipped (idempotent resumes).
        if [[ -d "${RUN_SAVE}/score" ]] && [[ $(find "${RUN_SAVE}/score" -name "*.json" 2>/dev/null | wc -l) -gt 0 ]]; then
            echo "  [run_${RUN}] already complete → skip"
            continue
        fi

        echo "  [run_${RUN}] starting... $(date)"

        if [[ "$DRY_RUN" == "true" ]]; then
            echo "  [dry-run] bash ${TOOL_USE_DIR}/run_eval.sh ${MODEL_PATH} ${TP}"
            continue
        fi

        export BFCL_PROMPT_MODE="${PROMPT_MODES[$MODEL_NAME]:-tool}"
        echo "  [prompt_mode] ${BFCL_PROMPT_MODE}"

        # Free GPUs from a previous vLLM server before starting a new one.
        pkill -f "vllm serve" 2>/dev/null || true
        sleep 5
        for _wait in $(seq 1 10); do
            GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s}')
            if [[ "${GPU_USED:-99999}" -lt 1000 ]]; then
                break
            fi
            sleep 3
        done

        set +e
        bash "${TOOL_USE_DIR}/run_eval.sh" "${MODEL_PATH}" "${TP}" \
            > "${MODEL_SAVE_DIR}/run_${RUN}.log" 2>&1
        RC=$?
        set -e

        if [[ $RC -eq 0 ]]; then
            mkdir -p "${RUN_SAVE}"
            MODEL_SHORT="$(basename "${MODEL_PATH}")"
            for subdir in result score; do
                SRC="${BFCL_BASE}/${subdir}/${MODEL_SHORT}"
                if [[ -d "${SRC}" ]]; then
                    cp -r "${SRC}" "${RUN_SAVE}/${subdir}"
                fi
            done
            echo "  [run_${RUN}] complete → ${RUN_SAVE}/"
        else
            echo "  [run_${RUN}] failed (exit code ${RC})"
            FAILED+=("${MODEL_NAME}/run_${RUN}")
        fi
    done
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Finished: $(date)"
echo "  Results:  ${SAVE_DIR}/"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""; echo "  [failures]"
    for f in "${FAILED[@]}"; do echo "    - ${f}"; done
fi
echo "════════════════════════════════════════════════════════════════"

echo ""
echo "Building summary table..."
python3 - "${SAVE_DIR}" <<'PYEOF'
import sys, os, json, glob
from collections import defaultdict

save_dir = sys.argv[1]
results = defaultdict(lambda: defaultdict(list))

for model_dir in sorted(glob.glob(os.path.join(save_dir, "*"))):
    if not os.path.isdir(model_dir):
        continue
    model_name = os.path.basename(model_dir)
    for run_dir in sorted(glob.glob(os.path.join(model_dir, "run_*"))):
        score_dir = os.path.join(run_dir, "score")
        if not os.path.isdir(score_dir):
            continue
        for score_file in glob.glob(os.path.join(score_dir, "*.json")):
            cat_name = os.path.basename(score_file).replace(".json", "")
            try:
                with open(score_file) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    acc = data.get("accuracy", data.get("overall_accuracy", None))
                    if acc is not None:
                        results[model_name][cat_name].append(acc)
            except Exception:
                pass

if not results:
    print("  no result files found.")
    sys.exit(0)

summary = {}
for model_name in sorted(results.keys()):
    summary[model_name] = {}
    for cat, scores in sorted(results[model_name].items()):
        avg = sum(scores) / len(scores) if scores else 0
        summary[model_name][cat] = {"scores": scores, "avg": avg, "n_runs": len(scores)}

summary_path = os.path.join(save_dir, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  summary saved to: {summary_path}")
PYEOF
