#!/usr/bin/env bash
# Environment bootstrap for BFCL tool-use eval.
# Delegates to the shared common_eval_env.sh and adds a PYTHONPATH entry for
# the vendored BFCL copy (if you have one under ./berkeley-function-call-leaderboard).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/eval_scripts/common_eval_env.sh"

# Prefer the local (patched) BFCL checkout if present.
LOCAL_BFCL="${SCRIPT_DIR}/berkeley-function-call-leaderboard"
if [[ -d "${LOCAL_BFCL}" ]]; then
    export PYTHONPATH="${LOCAL_BFCL}:${PYTHONPATH:-}"
fi
