#!/usr/bin/env bash
# Environment bootstrap for CURE coding eval (LiveBench / LiveCodeBench).
# Delegates to the shared common_eval_env.sh at the repo root.
set -euo pipefail

# Do not set SCRIPT_DIR here — callers (e.g. Coding/run_eval.sh) rely on
# their own SCRIPT_DIR and sourcing used to clobber it.
_CODING_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_CODING_ENV_DIR}/../eval_scripts/common_eval_env.sh"
