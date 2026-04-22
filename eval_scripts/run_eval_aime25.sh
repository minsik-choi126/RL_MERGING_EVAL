#!/usr/bin/env bash
# Run VERL validation-only evaluation on AIME25.
#
# Usage:
#   MODEL_PATH=/path/to/model bash run_eval_aime25.sh
#   (or called from eval_all.sh)

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_eval_env.sh"

# ── User-overridable configuration ──
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"

parse_model_dir_from_path() {
    local trimmed_path="${1%/}"
    if [[ "${trimmed_path}" =~ (Qwen3[^/[:space:]]*) ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    printf '%s\n' "${trimmed_path##*/}"
}

# Benchmark data.
AIME_VAL_PARQUET="${AIME_VAL_PARQUET:-${EVAL_DATA_ROOT}/aime25/test_verl_ready_with_instruction.parquet}"

# Output paths.
MODEL_DIR_NAME="${MODEL_DIR_NAME:-$(parse_model_dir_from_path "${MODEL_PATH}")}"
REPO_ROOT_GUESS="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT_GUESS}/results/verl_outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${MODEL_DIR_NAME}/evaluation_output/aime25}"
PROJECT_NAME="${PROJECT_NAME:-nemotron-cascade-parallel}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

# Wandb experiment name: {model}_{task}_{last_folder}
TASK_LABEL="aime25"
_BASENAME_OF_PATH="$(basename "${MODEL_PATH%/}")"
if [[ "${MODEL_DIR_NAME}" == "${_BASENAME_OF_PATH}" ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_DIR_NAME}_${TASK_LABEL}}"
else
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_DIR_NAME}_${TASK_LABEL}_${_BASENAME_OF_PATH}}"
fi

# Cluster parameters.
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-float16}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-sum-norm}"

mkdir -p "${OUTPUT_DIR}"

[[ -f "${AIME_VAL_PARQUET}" ]] || { echo "AIME25 parquet not found: ${AIME_VAL_PARQUET}" >&2; exit 1; }

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${AIME_VAL_PARQUET}" \
    data.val_files="${AIME_VAL_PARQUET}" \
    data.train_batch_size=1 \
    data.val_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-30720}" \
    data.filter_overlong_prompts=True \
    data.validation_shuffle=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.betas='[0.9,0.95]' \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.dtype="${DTYPE}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.95 \
    actor_rollout_ref.rollout.top_p=0.6 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.dtype="${DTYPE}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
    reward_manager.name=naive \
    reward_manager.source=register \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_only=True \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    +reward_model.reward_kwargs.overlong_filtering=False \
    hydra.run.dir="${OUTPUT_DIR}/hydra"
