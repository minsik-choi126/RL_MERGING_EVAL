#!/usr/bin/env bash
# Environment bootstrap for BFCL tool-use eval.
# Delegates to the shared common_eval_env.sh and adds a PYTHONPATH entry for
# the vendored BFCL copy (if you have one under ./berkeley-function-call-leaderboard).
set -euo pipefail

# Namespaced to avoid clobbering the caller's SCRIPT_DIR.
_TOOL_USE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_TOOL_USE_ENV_DIR}/../eval_scripts/common_eval_env.sh"

# Prefer the local (patched) BFCL checkout if present.
LOCAL_BFCL="${_TOOL_USE_ENV_DIR}/berkeley-function-call-leaderboard"
if [[ -d "${LOCAL_BFCL}" ]]; then
    export PYTHONPATH="${LOCAL_BFCL}:${PYTHONPATH:-}"
fi
