#!/usr/bin/env bash
# Environment bootstrap for MemAgent RULER eval.
# Delegates to the shared common_eval_env.sh and sets MemAgent-specific vars.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/eval_scripts/common_eval_env.sh"

# Data + port config for the MemAgent RULER tests.
export DATAROOT="${DATAROOT:-${SCRIPT_DIR}/hotpotqa}"
export SERVE_PORT="${SERVE_PORT:-8000}"
export DASH_PORT="${DASH_PORT:-8265}"

# MemAgent code (for python-path imports of `taskutils.memory_eval.utils.recurrent`).
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
