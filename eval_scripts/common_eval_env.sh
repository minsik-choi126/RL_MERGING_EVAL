#!/usr/bin/env bash
# Common environment bootstrap for VERL validation-only evaluation scripts
# (AIME24/25/26, IFEval). All cache / temp paths are env-var-driven so no
# personal paths leak through.
#
# Override any of the following before calling:
#   EVAL_VENV         — Python virtualenv to activate (default: ./.eval under repo)
#   NEMOTRON_REPO_ROOT— VERL repo root with custom reward functions
#   EVAL_DATA_ROOT    — Parquet data root (aime24/aime25/aime26, ifeval, ...)
#   CACHE_BASE        — Base dir for all caches (default: $HOME/.cache/rl_merging_eval)
#   TMPDIR/RAY_TMPDIR — LOCAL-disk temp dirs (default: /tmp/...-$USER)

set -euo pipefail

# ── Resolve paths ──
# Use a namespaced variable here; callers often have their own SCRIPT_DIR and
# sourcing this file used to clobber it.
_CEE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_CEE_DIR}/.." && pwd)"

# ── VERL repo root (custom AIME reward functions live here) ──
# If you have a local checkout, point NEMOTRON_REPO_ROOT at it. Otherwise the
# verl package installed via pip is used as-is.
export NEMOTRON_REPO_ROOT="${NEMOTRON_REPO_ROOT:-${REPO_ROOT}/external/verl_nemotron_merge}"

# ── Evaluation data root (parquet files for AIME, IFEval, …) ──
# Expected layout under EVAL_DATA_ROOT:
#   aime24/test_verl_ready_with_instruction.parquet
#   aime25/test_verl_ready_with_instruction.parquet
#   aime26/test_verl_ready_with_instruction.parquet
export EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-${REPO_ROOT}/data}"

# ── Activate eval virtualenv (falls back to system Python silently) ──
EXISTING_VENV="${EVAL_VENV:-${REPO_ROOT}/.eval}"
if [[ "${SKIP_EVAL_VENV:-0}" == "1" ]]; then
    echo "  [env] SKIP_EVAL_VENV=1 → using system Python"
elif [[ -f "${EXISTING_VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${EXISTING_VENV}/bin/activate"
    echo "  [env] venv activated: ${EXISTING_VENV}"
else
    echo "  [env] venv not found → using system Python (${EXISTING_VENV})"
fi

# ── Cache base (single knob controls every library cache location) ──
CACHE_BASE="${CACHE_BASE:-${HOME}/.cache/rl_merging_eval}"
mkdir -p "${CACHE_BASE}"

# TMPDIR / RAY_TMPDIR must be on LOCAL disk (not NFS) to avoid
# "Device or resource busy" errors from .nfs silly-rename files.
# Ray's AF_UNIX socket limits this to < 107 bytes, so keep it short.
export TMPDIR="${TMPDIR:-/tmp/eval_tmp_${USER:-$(id -un)}}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER:-$(id -un)}}"

# System-level redirects.
export XDG_CACHE_HOME="${CACHE_BASE}/xdg"
export PIP_CACHE_DIR="${CACHE_BASE}/pip"

# HuggingFace.
export HF_HOME="${CACHE_BASE}/huggingface"
export TRANSFORMERS_CACHE="${CACHE_BASE}/huggingface/hub"
export HF_DATASETS_CACHE="${CACHE_BASE}/huggingface/datasets"

# PyTorch.
export TORCH_HOME="${CACHE_BASE}/torch"
export TORCH_EXTENSIONS_DIR="${CACHE_BASE}/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_BASE}/torchinductor"

# Triton / CUDA.
export TRITON_CACHE_DIR="${CACHE_BASE}/triton"
export CUDA_CACHE_PATH="${CACHE_BASE}/cuda"

# vLLM — torch_compile_cache, deep_gemm JIT, all_reduce kernels.
export VLLM_CACHE_ROOT="${CACHE_BASE}/vllm"
export VLLM_CONFIG_ROOT="${CACHE_BASE}/vllm_config"
export DG_JIT_CACHE_DIR="${CACHE_BASE}/deep_gemm"

# FlashInfer — JIT-compiled CUDA kernels (can be hundreds of MBs).
export FLASHINFER_WORKSPACE_BASE="${CACHE_BASE}/flashinfer_base"

# Numba / NLTK / Matplotlib.
export NUMBA_CACHE_DIR="${CACHE_BASE}/numba"
export NLTK_DATA="${CACHE_BASE}/nltk_data"
export MPLCONFIGDIR="${CACHE_BASE}/matplotlib"

# Wandb — fully disabled by default (some setups crash on resource limits).
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DIR="${CACHE_BASE}/wandb"
export WANDB_CACHE_DIR="${CACHE_BASE}/wandb"
export WANDB_CONFIG_DIR="${CACHE_BASE}/wandb_config"
export WANDB_DATA_DIR="${CACHE_BASE}/wandb_data"

# Create required directories.
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" \
         "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" \
         "${TORCH_HOME}" "${TORCH_EXTENSIONS_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
         "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}" \
         "${VLLM_CACHE_ROOT}" "${VLLM_CONFIG_ROOT}" "${DG_JIT_CACHE_DIR}" \
         "${FLASHINFER_WORKSPACE_BASE}" "${NUMBA_CACHE_DIR}" "${NLTK_DATA}" \
         "${WANDB_DIR}" "${WANDB_CONFIG_DIR}" "${WANDB_DATA_DIR}" \
         "${MPLCONFIGDIR}"

# Ray: raise disk threshold + disable metrics (OpenTelemetry crashes).
export RAY_storage_monitor_threshold="${RAY_storage_monitor_threshold:-0.9999}"
export RAY_METRICS_EXPORT_PORT="${RAY_METRICS_EXPORT_PORT:-0}"
export RAY_enable_metrics_collection="${RAY_enable_metrics_collection:-0}"
# Limit simultaneous worker startup to avoid pthread_create exhaustion.
export RAY_maximum_startup_concurrency="${RAY_maximum_startup_concurrency:-8}"

# Make custom reward modules importable (VERL repo on PYTHONPATH).
if [[ -d "${NEMOTRON_REPO_ROOT}" ]]; then
    export PYTHONPATH="${NEMOTRON_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

# Thread / resource limits — reduce intra-op parallelism to avoid
# "can't start new thread" errors on nodes with low vm.max_map_count.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# Logging verbosity.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning:megatron}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

# Prevent stray __pycache__ writes to read-only mounts.
export PYTHONDONTWRITEBYTECODE=1
