#!/usr/bin/env bash
# run_eval.sh — unified evaluation entry point.
#
# Loads config.env (if present), activates the virtualenv configured in
# common_eval_env.sh, then delegates to unified_eval.py.
#
# If setup has never been run (no .setup_done sentinel), this script aborts
# with a hint. Pass --auto-setup to run setup.sh inline before evaluating.
#
# Usage:
#   bash run_eval.sh --model /path/to/model --benchmarks all
#   bash run_eval.sh --model /path/to/model --benchmarks aime24 ifeval --tp 2
#   bash run_eval.sh --auto-setup --model /path/to/model --benchmarks all
#
# All remaining arguments are forwarded to unified_eval.py
# (run `python unified_eval.py --help` for the full option list).

set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse our own --auto-setup flag; forward everything else.
AUTO_SETUP=0
FORWARD=()
for arg in "$@"; do
    if [[ "$arg" == "--auto-setup" ]]; then
        AUTO_SETUP=1
    else
        FORWARD+=("$arg")
    fi
done

# Load user env overrides.
if [[ -f "${EVAL_DIR}/config.env" ]]; then
    set -a; source "${EVAL_DIR}/config.env"; set +a
fi

# First-run check.
if [[ ! -f "${EVAL_DIR}/.setup_done" ]]; then
    if [[ "${AUTO_SETUP}" -eq 1 ]]; then
        echo "[run_eval] .setup_done missing → invoking setup.sh"
        bash "${EVAL_DIR}/setup.sh"
    else
        echo "[run_eval] setup.sh has not been run yet." >&2
        echo "           → bash setup.sh                 (one-time, ~10-30 min)" >&2
        echo "           → or: bash run_eval.sh --auto-setup <args...>" >&2
        exit 2
    fi
fi

# Activate venv + env.
# shellcheck disable=SC1091
source "${EVAL_DIR}/eval_scripts/common_eval_env.sh"

exec python "${EVAL_DIR}/unified_eval.py" "${FORWARD[@]}"
