#!/bin/bash
set -euo pipefail

###############################################################################
# run_eval.sh — 모델 경로만 넣으면 MemAgent RULER 평가 실행
#
# Usage:
#   bash run_eval.sh <model_path> [model_path2 ...] [OPTIONS]
#
# Examples:
#   bash run_eval.sh /path/to/model
#   bash run_eval.sh /path/a /path/b --tests hqa
#   bash run_eval.sh /path/model --method openai --tp 2
#   bash run_eval.sh /path/model --force
#
# Options:
#   --tests       hqa | ruler | all       (default: all)
#   --method      openai | recurrent      (default: recurrent)
#                 recurrent 시 rope_scaling(YaRN)이 자동으로 temp dir에 주입됨
#   --tp N        tensor parallel size    (default: 1)
#   --force       결과 있어도 강제 재실행
#   --dry-run     커맨드만 출력, 실행 X
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PY="${SCRIPT_DIR}/taskutils/memory_eval/run.py"

# ─── 기본값 ───────────────────────────────────────────────────────────────
TESTS="all"
METHOD="recurrent"
TP=1
FORCE=false
DRY_RUN=false
MODELS=()

# ─── Arg 파싱 ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tests)       TESTS="$2";   shift 2 ;;
        --method)      METHOD="$2";  shift 2 ;;
        --tp)          TP="$2";      shift 2 ;;
        --force)       FORCE=true;   shift ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --help|-h)
            sed -n '/^# Usage:/,/^###/p' "$0" | head -n -1 | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        -*)
            echo "[ERROR] 알 수 없는 옵션: $1" >&2; exit 1
            ;;
        *)
            MODELS+=("$1"); shift
            ;;
    esac
done

# ─── 검증 ─────────────────────────────────────────────────────────────────
if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "[ERROR] 모델 경로를 하나 이상 지정하세요." >&2
    echo "  Usage: bash run_eval.sh <model_path> [OPTIONS]" >&2
    exit 1
fi
if [[ "$TESTS" != "hqa" && "$TESTS" != "ruler" && "$TESTS" != "all" ]]; then
    echo "[ERROR] --tests 는 hqa | ruler | all 중 하나" >&2; exit 1
fi
if [[ "$METHOD" != "openai" && "$METHOD" != "recurrent" ]]; then
    echo "[ERROR] --method 는 openai | recurrent 중 하나" >&2; exit 1
fi

# ─── 환경 로드 ────────────────────────────────────────────────────────────
source "${SCRIPT_DIR}/env.sh"
export DATAROOT="${SCRIPT_DIR}/hotpotqa"

# ─── 메인 ─────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo "  MemAgent Evaluation"
echo "  모델 수  : ${#MODELS[@]}"
echo "  tests    : ${TESTS}"
echo "  method   : ${METHOD}"
echo "  tp       : ${TP}"
echo "  force    : ${FORCE}"
echo "  dry-run  : ${DRY_RUN}"
echo "  시작     : $(date)"
echo "════════════════════════════════════════════════════════════════"

FAILED=()

for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME="$(basename "${MODEL_PATH%/}")"

    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  모델: ${MODEL_NAME}"
    echo "  경로: ${MODEL_PATH}"
    echo "────────────────────────────────────────────────────────────────"

    if [[ ! -d "${MODEL_PATH}" ]]; then
        echo "  [ERROR] 디렉토리 없음 → 건너뜀" >&2
        FAILED+=("${MODEL_NAME} (경로 없음)")
        continue
    fi

    # 평가 실행 (recurrent 모드 시 rope_scaling은 run.py가 temp dir에 자동 주입)
    EXTRA=""
    [[ "$FORCE" == "true" ]] && EXTRA="--force"

    EVAL_CMD="python ${RUN_PY} \
        --model ${MODEL_PATH} \
        --name ${MODEL_NAME} \
        --tp ${TP} \
        --method ${METHOD} \
        --tests ${TESTS} \
        ${EXTRA}"

    echo "  [eval] ${EVAL_CMD}"
    if [[ "$DRY_RUN" == "false" ]]; then
        if ! eval "${EVAL_CMD}"; then
            echo "  [ERROR] 평가 실패: ${MODEL_NAME}" >&2
            FAILED+=("${MODEL_NAME} (평가 실패)")
        fi
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  완료: $(date)"
echo "  결과: ${SCRIPT_DIR}/taskutils/memory_eval/results/"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  [실패]"
    for f in "${FAILED[@]}"; do echo "    - ${f}"; done
    echo "════════════════════════════════════════════════════════════════"
    exit 1
fi
echo "════════════════════════════════════════════════════════════════"
