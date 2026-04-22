#!/usr/bin/env bash
# setup.sh — one-shot environment bootstrap.
#
# Idempotent: re-running skips work that is already done.
#
# What this script does:
#   1. Create a Python venv at $EVAL_VENV (default: ./.eval) and activate it.
#   2. Install pip dependencies from requirements-eval.txt plus vLLM and flash-attn.
#   3. Install BFCL (bfcl-eval) for the tool_use benchmark.
#   4. Clone VERL into ./external/verl (shallow) for AIME / IFEval.
#   5. Download AIME 24/25/26 into $EVAL_DATA_ROOT (default: ./data) as
#      VERL-ready parquets via scripts/download_data.py.
#   6. Download MemAgent RULER-HQA JSON into ./MemAgent/hotpotqa.
#
# Flags:
#   --skip-venv      reuse current interpreter instead of making a venv
#   --skip-pip       skip pip installs
#   --skip-vllm      skip vLLM + flash-attn (heavy)
#   --skip-bfcl      skip bfcl-eval install
#   --skip-verl      skip VERL clone
#   --skip-data      skip dataset downloads
#   --force-data     re-download datasets even if already present

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SKIP_VENV=0
SKIP_PIP=0
SKIP_VLLM=0
SKIP_BFCL=0
SKIP_VERL=0
SKIP_DATA=0
FORCE_DATA=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-venv)   SKIP_VENV=1;  shift ;;
        --skip-pip)    SKIP_PIP=1;   shift ;;
        --skip-vllm)   SKIP_VLLM=1;  shift ;;
        --skip-bfcl)   SKIP_BFCL=1;  shift ;;
        --skip-verl)   SKIP_VERL=1;  shift ;;
        --skip-data)   SKIP_DATA=1;  shift ;;
        --force-data)  FORCE_DATA=1; shift ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "[ERROR] unknown flag: $1" >&2; exit 1 ;;
    esac
done

# Load user overrides if present.
if [[ -f "${SCRIPT_DIR}/config.env" ]]; then
    set -a; source "${SCRIPT_DIR}/config.env"; set +a
fi

EVAL_VENV="${EVAL_VENV:-${SCRIPT_DIR}/.eval}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-${SCRIPT_DIR}/data}"
NEMOTRON_REPO_ROOT="${NEMOTRON_REPO_ROOT:-${SCRIPT_DIR}/external/verl}"

echo "============================================================"
echo "  setup.sh"
echo "  venv : ${EVAL_VENV}"
echo "  data : ${EVAL_DATA_ROOT}"
echo "  verl : ${NEMOTRON_REPO_ROOT}"
echo "============================================================"

# ── 1) venv ────────────────────────────────────────────────────────────────
if [[ "${SKIP_VENV}" -eq 0 ]]; then
    if [[ ! -d "${EVAL_VENV}" ]]; then
        echo "[venv] creating ${EVAL_VENV}"
        python3 -m venv "${EVAL_VENV}"
    fi
    # shellcheck disable=SC1091
    source "${EVAL_VENV}/bin/activate"
    python -m pip install --upgrade pip wheel setuptools >/dev/null
fi

# ── 2) pip deps ────────────────────────────────────────────────────────────
if [[ "${SKIP_PIP}" -eq 0 ]]; then
    echo "[pip] installing glue dependencies"
    pip install -r "${SCRIPT_DIR}/requirements-eval.txt"
fi

# ── 3) vLLM + flash-attn (heavy) ───────────────────────────────────────────
if [[ "${SKIP_VLLM}" -eq 0 ]]; then
    if ! python -c "import vllm" 2>/dev/null; then
        echo "[vllm] installing vllm (this takes a while on first run)"
        pip install "vllm>=0.7.0"
    else
        echo "[vllm] already installed ($(python -c 'import vllm;print(vllm.__version__)'))"
    fi
    if ! python -c "import flash_attn" 2>/dev/null; then
        echo "[flash-attn] installing (needs a CUDA toolchain)"
        pip install flash-attn --no-build-isolation || \
            echo "[flash-attn] install failed — you can retry later; not fatal"
    else
        echo "[flash-attn] already installed"
    fi
fi

# ── 4) BFCL ────────────────────────────────────────────────────────────────
if [[ "${SKIP_BFCL}" -eq 0 ]]; then
    if ! python -c "import bfcl_eval" 2>/dev/null && ! command -v bfcl &>/dev/null; then
        echo "[bfcl] installing bfcl-eval"
        pip install bfcl-eval
    else
        echo "[bfcl] already installed"
    fi

    # Patch the local_inference handler with our fixes (tools block, dtype,
    # <tool_call> XML parsing) so RL-finetuned tool-calling models score fairly.
    HANDLER_SRC="${SCRIPT_DIR}/Tool_use/handlers/qwen_generic.py"
    if [[ -f "${HANDLER_SRC}" ]]; then
        DEST_DIR="$(python -c 'import bfcl.model_handler.local_inference as m, os; print(os.path.dirname(m.__file__))' 2>/dev/null || true)"
        if [[ -n "${DEST_DIR}" && -d "${DEST_DIR}" ]]; then
            cp -f "${HANDLER_SRC}" "${DEST_DIR}/qwen_generic.py"
            echo "[bfcl] patched handler → ${DEST_DIR}/qwen_generic.py"
        fi
    fi
fi

# ── 5) VERL clone ──────────────────────────────────────────────────────────
if [[ "${SKIP_VERL}" -eq 0 ]]; then
    if [[ ! -d "${NEMOTRON_REPO_ROOT}/.git" ]]; then
        echo "[verl] cloning volcengine/verl → ${NEMOTRON_REPO_ROOT}"
        mkdir -p "$(dirname "${NEMOTRON_REPO_ROOT}")"
        git clone --depth 1 https://github.com/volcengine/verl.git "${NEMOTRON_REPO_ROOT}"
        echo "[verl] NOTE: AIME custom reward files (aime.py / grader.py / naive.py /"
        echo "       reward_score/__init__.py) are NOT in upstream. Drop your custom"
        echo "       versions into ${NEMOTRON_REPO_ROOT}/verl/ before running AIME."
    else
        echo "[verl] already cloned at ${NEMOTRON_REPO_ROOT}"
    fi
    pip install -e "${NEMOTRON_REPO_ROOT}" || \
        echo "[verl] editable install failed — continuing (verl may also work from PYTHONPATH)"
fi

# ── 6) Datasets ────────────────────────────────────────────────────────────
if [[ "${SKIP_DATA}" -eq 0 ]]; then
    FORCE_FLAG=""
    [[ "${FORCE_DATA}" -eq 1 ]] && FORCE_FLAG="--force"
    EVAL_DATA_ROOT="${EVAL_DATA_ROOT}" python "${SCRIPT_DIR}/scripts/download_data.py" \
        --data-root "${EVAL_DATA_ROOT}" \
        --memagent-root "${SCRIPT_DIR}/MemAgent" \
        ${FORCE_FLAG}
fi

# ── Final sentinel ─────────────────────────────────────────────────────────
touch "${SCRIPT_DIR}/.setup_done"
echo
echo "============================================================"
echo "  setup complete"
echo "============================================================"
echo "  Next:"
echo "    bash run_eval.sh --model /path/to/model --benchmarks all"
echo
echo "  If AIME reports 0% accuracy, you're missing the custom VERL reward"
echo "  files — see docs/EVAL_GUIDE.md §Known Pitfalls #1."
