#!/usr/bin/env bash
# ============================================================================
# run_eval.sh -- Unified evaluation entry point
#
# Activates the eval virtualenv, sets environment, then delegates to
# unified_eval.py which orchestrates all benchmarks.
#
# Usage:
#   bash run_eval.sh --model /path/to/model --benchmarks all
#   bash run_eval.sh --model /path/to/model --benchmarks aime24 aime25 ifeval
#   bash run_eval.sh --model /path/to/model --benchmarks coding --gpu_per_engine 4
#   bash run_eval.sh --model /path/to/model --benchmarks memagent --tp 1
#
# All arguments are forwarded to unified_eval.py. Run with --help for full options.
# ============================================================================

set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Activate venv and set environment ──
source "${EVAL_DIR}/eval_scripts/common_eval_env.sh"

# ── Forward all arguments to the Python orchestrator ──
exec python "${EVAL_DIR}/unified_eval.py" "$@"
