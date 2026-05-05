#!/usr/bin/env bash
# Merge ours (W_expert_top<pct>) + 9 baseline methods on Qwen3-1.7B + 3 RL experts.
#
# All merges land under: $OUT_DIR/<method>/  (default: KT_merge/outputs/merges/)
#
# Methods (10 total):
#   ours: W_expert_top<pct>
#   baselines: task_arithmetic, ties, dare_ta, star, tsv,
#              iso_c, iso_cts, ram, ram_plus
#
# Prereqs:
#   - models/{ifeval,math,lucy}/ symlinks present
#   - data/per_query/{ifeval,math,lucy}.npz present (run prep_proxy_qwen3.py)
#   - outputs/W_expert_top<pct>_perexpert.npz present (compute_W_expert.py)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HERE%/scripts}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
ENERGY="${ENERGY:-0.90}"
DEVICE="${DEVICE:-cuda:0}"
TIES_DEVICE="${TIES_DEVICE:-${DEVICE}}"   # set to "cpu" to avoid OOM on TIES / DARE_TIES (both flatten all params)
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/merges}"
LOG_DIR="${LOG_DIR:-${ROOT}/outputs/merge_logs}"

IFEVAL="${ROOT}/models/ifeval"
MATH="${ROOT}/models/math"
LUCY="${ROOT}/models/lucy"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

for d in "${IFEVAL}" "${MATH}" "${LUCY}"; do
    if ! ls "$d"/*.safetensors >/dev/null 2>&1; then
        echo "[ERR] missing safetensors in: $d" >&2; exit 1
    fi
done

# ── Helper ──────────────────────────────────────────────────────────────────
run() {  # run <tag> <python_command...>
    local tag="$1"; shift
    local save="${OUT_DIR}/${tag}"
    local log="${LOG_DIR}/${tag}.log"
    if ls "${save}"/*.safetensors >/dev/null 2>&1; then
        echo "  [SKIP] ${tag} (safetensors exists)"
        return 0
    fi
    mkdir -p "${save}"
    echo ""
    echo "── merge: ${tag} ──"
    echo "  log: ${log}"
    set +e
    "$@" 2>&1 | tee "${log}"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "${rc}" -ne 0 ]; then
        echo "  [ERR] ${tag} failed (rc=${rc})"
        return 1
    fi
}

# ── Ours ────────────────────────────────────────────────────────────────────
# Variant ktcol_polar_renorm with column-side W from abs+soft |Δlogp|^α
# weighting on every answer-token, multiplied by ‖W[:,c]‖₂ — headline method.
W_COL_FILE="${W_COL_FILE:-${ROOT}/outputs/W_col_abs_perexpert.npz}"
if [ ! -f "${W_COL_FILE}" ]; then
    echo "[ERR] ${W_COL_FILE} missing — run compute_W_col.py first" >&2
    exit 1
fi
OURS_TAG="${OURS_TAG:-W_col_abs}"
ABL_ROOT="${OUT_DIR}/_ablation/${OURS_TAG}"
ABL_OUT="${ABL_ROOT}/ktcol_polar_renorm"
OURS_TARGET="${OUT_DIR}/${OURS_TAG}"

# Detect per-expert (2D) vs union (1D) W just for logging
W_NDIM=$(python3 -c "import numpy as np; a=np.load('${W_COL_FILE}'); k=list(a.keys())[0]; print(a[k].ndim)" 2>/dev/null || echo 1)
echo "[info] W_col file is ${W_NDIM}D  (1=union, 2=per-expert)"

OURS_READY=0
if ls "${OURS_TARGET}"/*.safetensors >/dev/null 2>&1; then
    echo "  [SKIP] ${OURS_TAG} (safetensors exists)"
    OURS_READY=1
elif ! ls "${ABL_OUT}"/*.safetensors >/dev/null 2>&1; then
    log="${LOG_DIR}/${OURS_TAG}.log"
    echo ""
    echo "── merge: ${OURS_TAG} ──"
    echo "  log: ${log}"
    set +e
    python "${HERE}/merge_ablation.py" \
        --variants ktcol_polar_renorm \
        --w_col_file "${W_COL_FILE}" \
        --energy "${ENERGY}" \
        --device "${DEVICE}" \
        --out_root "${ABL_ROOT}" \
        --base_model "${BASE_MODEL}" \
        --expert_paths "ifeval=${IFEVAL}" "math=${MATH}" "lucy=${LUCY}" \
        2>&1 | tee "${log}"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "${rc}" -ne 0 ]; then
        echo "  [ERR] ${OURS_TAG} failed (rc=${rc})"
        exit 1
    fi
fi

if [ "${OURS_READY}" != "1" ]; then
    if ! ls "${ABL_OUT}"/*.safetensors >/dev/null 2>&1; then
        echo "[ERR] ${ABL_OUT} has no safetensors after merge" >&2
        exit 1
    fi
    if [ -d "${OURS_TARGET}" ] && [ ! -L "${OURS_TARGET}" ]; then
        if [ -z "$(find "${OURS_TARGET}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
            rmdir "${OURS_TARGET}"
        else
            echo "[ERR] ${OURS_TARGET} exists but is not an empty directory or symlink" >&2
            exit 1
        fi
    fi
    ln -sfnT "${ABL_OUT}" "${OURS_TARGET}"
    echo "  symlinked ${ABL_OUT}  →  ${OURS_TARGET}"
else
    if [ -L "${OURS_TARGET}" ]; then
        echo "  symlink ready: ${OURS_TARGET}"
    else
        echo "  output ready: ${OURS_TARGET}"
    fi
fi

# ── Baselines (skip with SKIP_BASELINES=1 to run only ours) ────────────────
FAILED=()
if [ "${SKIP_BASELINES:-0}" = "1" ]; then
    echo ""
    echo "── Baselines: SKIPPED (SKIP_BASELINES=1)"
else
    BASELINES=(
        "task_arithmetic     --method task_arithmetic"
        "ties                --method ties --device ${TIES_DEVICE}"
        "dare_ta             --method dare --dare_merge_method task_arithmetic --device ${DEVICE}"
        "star                --method star --device ${DEVICE}"
        "tsv                 --method tsv --device ${DEVICE}"
        "iso_c               --method iso_c --device ${DEVICE}"
        "iso_cts             --method iso_cts --device ${DEVICE}"
        "ram                 --method ram --device ${DEVICE}"
        "ram_plus            --method ram_plus --device ${DEVICE}"
    )
    for spec in "${BASELINES[@]}"; do
        tag=$(echo "${spec}" | awk '{print $1}')
        extra=$(echo "${spec}" | cut -d' ' -f2-)
        set +e
        run "${tag}" \
            python "${HERE}/merge.py" \
                ${extra} \
                --base_model "${BASE_MODEL}" \
                --expert_models "${IFEVAL}" "${MATH}" "${LUCY}" \
                --save_dir "${OUT_DIR}/${tag}"
        rc=$?
        set -e
        [ "${rc}" -ne 0 ] && FAILED+=("${tag}")
    done
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Merging Summary"
echo "════════════════════════════════════════════════════════════════"
echo "  out_dir: ${OUT_DIR}"
ls "${OUT_DIR}/" 2>/dev/null
echo ""
echo "  failed: ${#FAILED[@]} → ${FAILED[*]:-(none)}"
[ "${#FAILED[@]}" -gt 0 ] && exit 1
exit 0
