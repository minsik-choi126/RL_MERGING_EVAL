#!/usr/bin/env bash
# Environment bootstrap for CURE coding eval (LiveBench / LiveCodeBench).
# Delegates to the shared common_eval_env.sh at the repo root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/eval_scripts/common_eval_env.sh"
