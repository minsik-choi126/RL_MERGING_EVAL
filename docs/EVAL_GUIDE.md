# Unified Evaluation Guide

## Quick Start

```bash
# Run ALL benchmarks
bash run_eval.sh --model /path/to/model --benchmarks all

# AIME24/25/26 개별 실행
bash run_eval.sh --model /path/to/model --benchmarks aime24
bash run_eval.sh --model /path/to/model --benchmarks aime25
bash run_eval.sh --model /path/to/model --benchmarks aime26
bash run_eval.sh --model /path/to/model --benchmarks aime24 aime25 aime26   # 전부

# IFEval만
bash run_eval.sh --model /path/to/model --benchmarks ifeval --tp 2

# LiveBench / LiveCodeBench 개별 실행
bash run_eval.sh --model /path/to/model --benchmarks livebench --gpu_per_engine 2
bash run_eval.sh --model /path/to/model --benchmarks livecodebench --gpu_per_engine 2
bash run_eval.sh --model /path/to/model --benchmarks coding --gpu_per_engine 4  # 둘 다 (alias)

# Tool Use (BFCL)
bash run_eval.sh --model /path/to/model --benchmarks tool_use --tp 2

# MemAgent
bash run_eval.sh --model /path/to/model --benchmarks memagent --tp 1 --memagent_tests hqa

# 여러 벤치 조합
bash run_eval.sh --model /path/to/model --benchmarks aime25 aime26 ifeval livebench --tp 2 --n_gpus 2

# 여러 모델 루프
for model in /path/to/Model-A /path/to/Model-B; do
    bash run_eval.sh --model "$model" --benchmarks aime25 ifeval --tp 2 --n_gpus 2
done

# Dry run (명령어만 출력, 실행 안 함)
bash run_eval.sh --model /path/to/model --benchmarks all --dry_run
```

You can also call the Python script directly (if the venv and env vars are already active):

```bash
python unified_eval.py --model /path/to/model --benchmarks all
```


## Supported Benchmarks

| Benchmark       | Alias(es)    | What It Measures                      | Framework            |
|-----------------|--------------|---------------------------------------|----------------------|
| `aime24`        |              | AIME 2024 math reasoning              | VERL (verl.trainer)  |
| `aime25`        |              | AIME 2025 math reasoning              | VERL (verl.trainer)  |
| `aime26`        |              | AIME 2026 math reasoning              | VERL (verl.trainer)  |
| `ifeval`        |              | Instruction following (IFEval)        | vLLM direct (custom) |
| `livebench`     | `coding`     | LiveBench code generation             | vLLM multi-engine    |
| `livecodebench` | `coding`     | LiveCodeBench code generation         | vLLM multi-engine    |
| `tool_use`      |              | BFCL function calling (live+non_live) | BFCL CLI + vLLM      |
| `memagent`      |              | RULER HQA long-context memory         | vLLM serve + API     |

Special aliases:
- `all` -- runs every benchmark above
- `coding` -- runs `livebench` + `livecodebench`


## CLI Options

| Option              | Default   | Description                                              |
|---------------------|-----------|----------------------------------------------------------|
| `--model`           | (required)| Path to model directory                                  |
| `--benchmarks`      | `all`     | Space-separated list of benchmarks or aliases             |
| `--tp`              | `4`       | Tensor parallel size (for IFEval, Tool Use, MemAgent)     |
| `--n_gpus`          | auto      | Total GPUs for VERL AIME evals (0 = auto-detect)         |
| `--gpu_per_engine`  | `1`       | GPUs per vLLM engine for coding (1 for 1.7B, 4 for 7B)   |
| `--run`             | auto      | Run number (auto-increments from existing run_NNN dirs)   |
| `--output_dir`      | `results/`| Override base results directory                           |
| `--temperature`     | (default) | Override temperature for coding benchmarks                |
| `--chat_template`   | (default) | Chat template: `qwen` or `cure`                          |
| `--memagent_tests`  | `hqa`     | MemAgent test set: `hqa`, `ruler`, or `all`               |
| `--memagent_method` | `openai`  | MemAgent method: `openai` or `recurrent`                  |
| `--dry_run`         | off       | Print commands without executing                          |
| `--verbose`         | off       | Verbose output                                            |


## Model Compatibility

### Auto-Detection

The system reads `config.json` from the model directory to detect:
- **max_position_embeddings** -- used to auto-adjust AIME max_response_length
- **rope_scaling** -- detects YaRN/dynamic NTK for long-context models
- **num_attention_heads / num_key_value_heads** -- useful for TP constraint validation

### AIME max_response_length Calculation

```
max_response_length = min(30720, max_position_embeddings - 2048)
```

For a model with `max_position_embeddings=32768`, this gives 30720.
For a model with `max_position_embeddings=8192`, this gives 6144.

### Tensor Parallelism Constraints

TP size must evenly divide both `num_attention_heads` and `num_key_value_heads`:

| Model          | heads | kv_heads | Valid TP sizes    |
|----------------|-------|----------|-------------------|
| Qwen3-1.7B    | 16    | 8        | 1, 2, 4, 8       |
| Qwen2.5-7B    | 28    | 4        | 1, 2, 4           |

### rope_scaling

Models with `rope_scaling` in their config (e.g., YaRN) support longer contexts
than `max_position_embeddings` alone suggests. The MemAgent `recurrent` method
auto-injects rope_scaling into a temp config directory.


## GPU Requirements

| Benchmark   | Recommended GPUs | Notes                                       |
|-------------|------------------|---------------------------------------------|
| AIME24/25/26| 4                | VERL uses all GPUs via `n_gpus_per_node`    |
| IFEval      | 4 (TP=4)         | Single vLLM instance                        |
| LiveBench   | 4                | `gpu_per_engine=1`: 4 engines; `=4`: 1 engine |
| LiveCodeBench | 4              | Same as LiveBench                            |
| Tool Use    | 4 (TP=4)         | Single vLLM backend for BFCL                |
| MemAgent    | 1-4              | `vllm serve` with specified TP               |

For 1.7B models: `--tp 1 --gpu_per_engine 1` (uses 1 GPU per engine, 4 engines)
For 7B models: `--tp 4 --gpu_per_engine 4` (uses 4 GPUs per engine, 1 engine)


## Expected Runtimes

Approximate runtimes on 4x A100/H100:

| Benchmark     | 1.7B Model | 7B Model  |
|---------------|------------|-----------|
| AIME24        | ~15 min    | ~30 min   |
| AIME25        | ~15 min    | ~30 min   |
| AIME26        | ~15 min    | ~30 min   |
| IFEval        | ~5 min     | ~10 min   |
| LiveBench     | ~10 min    | ~20 min   |
| LiveCodeBench | ~10 min    | ~20 min   |
| Tool Use      | ~20 min    | ~40 min   |
| MemAgent      | ~15 min    | ~30 min   |
| **All**       | ~1.5 hr    | ~3 hr     |

Runtimes vary significantly based on hardware, model size, and context lengths.


## Output Format and Results

### Directory Structure

```
evaluation/results/
  run_001/
    eval_config.json        # Exact config used (for reproducibility)
    eval.log                # Full log (from run_eval.sh)
    summary.json            # Per-benchmark status and timing
    Coding/
      <model_short>.txt     # LiveBench + LiveCodeBench combined
    Tool_use/
      <model_short>/        # BFCL score JSONs
    MemAgent/
      ruler_hqa_<dist>/
        <model_short>.jsonl # HQA results per distance
  run_002/
    ...
```

### AIME Results

AIME results are stored in the VERL output directory:
- On DDN: `$OUTPUT_ROOT/<model>/evaluation_output/aime24/`
- Fallback: `evaluation/results/verl_outputs/<model>/evaluation_output/aime24/`

### IFEval Results

- Summary: `<output_root>/<model>/evaluation_output/ifeval/summary.json`
- Per-example: `<output_root>/<model>/evaluation_output/ifeval/results.jsonl`

Metrics:
- `prompt_level_strict_acc` -- fraction of prompts with ALL instructions followed
- `inst_level_strict_acc` -- fraction of individual instructions followed

### Coding Results

Result text files contain accuracy and unit test pass rates:
- LiveBench: accuracy (ACC) and unit test pass rate (UT)
- LiveCodeBench: accuracy (ACC) and unit test pass rate (UT)

### Tool Use Results

BFCL scores are JSON files per category (live_parallel, live_parallel_multiple, etc.).
Use `Tool_use/summarize_results.py` for aggregated scores.

### MemAgent Results

JSONL files per context distance (7K, 14K, 32K, 64K) with HotpotQA scores.


## Environment

### Virtual Environment

The evaluation venv is at `$REPO_ROOT/.eval`. It is activated
automatically by `run_eval.sh` via `common_eval_env.sh`.

Override with: `EVAL_VENV=/path/to/venv bash run_eval.sh ...`

### Key Paths

| What                  | Path                                           |
|-----------------------|------------------------------------------------|
| Eval venv             | `$REPO_ROOT/.eval`                   |
| VERL repo             | `$NEMOTRON_REPO_ROOT`             |
| Eval data (parquets)  | `$EVAL_DATA_ROOT/`                 |
| Cache base            | `$HOME/.cache/rl_merging_eval` or `$CACHE_BASE` override |
| TMPDIR / RAY_TMPDIR   | `/tmp/eval_tmp_<user>`, `/tmp/ray_<user>`      |
| Results               | `evaluation/results/run_NNN/`                  |

### Cache Environment Variables

`common_eval_env.sh` sets 20+ cache environment variables to avoid writing to
home directories or NFS. Key ones:
- `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE` -- HuggingFace
- `TORCH_HOME`, `TORCH_EXTENSIONS_DIR` -- PyTorch
- `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH` -- GPU kernels
- `VLLM_CACHE_ROOT` -- vLLM compile cache
- `TMPDIR`, `RAY_TMPDIR` -- temp files (LOCAL disk, not NFS)


## Troubleshooting

### NFS "Device or resource busy" / .nfs Silly-Rename

**Symptom:** `OSError: [Errno 16] Device or resource busy` during Ray/vLLM cleanup.

**Cause:** Ray temp files on NFS. When processes delete files that others still
have open, NFS renames them to `.nfsXXXX` files instead of deleting.

**Fix:** `TMPDIR` and `RAY_TMPDIR` are set to `/tmp/` (local disk) by
`common_eval_env.sh`. If the error persists, manually clean up:
```bash
ray stop --force
rm -rf /tmp/ray_* /tmp/eval_tmp_*
```

### flash_attn / FlashInfer Errors

**Symptom:** `ModuleNotFoundError: No module named 'flash_attn'` or
FlashInfer JIT compilation fails.

**Fix:** Ensure the eval venv has flash-attn installed:
```bash
source $REPO_ROOT/.eval/bin/activate
pip install flash-attn --no-build-isolation
```

FlashInfer JIT cache is at `$FLASHINFER_WORKSPACE_BASE`. Clear it if corrupted:
```bash
rm -rf $HOME/.cache/rl_merging_eval/flashinfer_base
```

### vLLM Version Mismatch

**Symptom:** `TypeError` or unexpected keyword arguments in vLLM calls.

**Fix:** The eval scripts are tested with vLLM v0.7+. Check version:
```bash
python -c "import vllm; print(vllm.__version__)"
```

### Ray "Can't start new thread" / pthread_create

**Symptom:** `RuntimeError: can't start new thread` during VERL evaluation.

**Cause:** Low `vm.max_map_count` or too many Ray workers.

**Fix:** `common_eval_env.sh` sets `RAY_maximum_startup_concurrency=8` and
`MALLOC_ARENA_MAX=2` to mitigate this. If it persists, try reducing:
```bash
export RAY_maximum_startup_concurrency=4
export OMP_NUM_THREADS=1
```

### AIME: "max_model_len is too large"

**Symptom:** vLLM rejects `max_model_len` exceeding the model's
`max_position_embeddings`.

**Fix:** `unified_eval.py` auto-detects `max_position_embeddings` from
`config.json` and sets `max_response_length = min(30720, max_pos - 2048)`.
If the model has `rope_scaling`, vLLM may accept larger values. Override with:
```bash
MAX_RESPONSE_LENGTH=8192 bash run_eval.sh --model /path --benchmarks aime24
```

### IFEval: TP Not Applied

**Symptom:** IFEval runs on fewer GPUs than expected.

**Fix:** The unified system passes `--tp` correctly to `ifeval_eval.py`.
Verify with `--dry_run`:
```bash
bash run_eval.sh --model /path --benchmarks ifeval --tp 4 --dry_run
```

### CUDA Out of Memory

**Symptom:** `torch.cuda.OutOfMemoryError` during evaluation.

**Fix:**
- For coding: increase `--gpu_per_engine` (e.g., 4 for 7B models)
- For AIME: reduce `MAX_RESPONSE_LENGTH` or use more GPUs
- For MemAgent: increase `--tp`
- General: `gpu_memory_utilization` is set to 0.8-0.9 in the sub-scripts

### Wandb Errors

Wandb is disabled by default (`WANDB_MODE=disabled`). If you need wandb logging,
set `WANDB_MODE=online` before running.


## Known Pitfalls (과거 실수 기록)

### 1. VERL upstream merge로 커스텀 reward 함수 유실

**증상:** AIME24/25 평가 시 모든 문제에서 acc=0%, reward=-1.0

**원인:** VERL repo(`$NEMOTRON_REPO_ROOT`)에서 upstream merge 후
커스텀 파일들이 덮어씌워짐:
- `verl/utils/reward_score/aime.py` — AIME 전용 reward (boxed, bold 등 5가지 패턴 추출)
- `verl/utils/reward_score/grader.py` — math_equal 비교 로직
- `verl/utils/reward_score/__init__.py` — aime data_source 라우팅
- `verl/workers/reward_manager/naive.py` — overlong_filtering 파라미터

upstream의 `math_dapo.py`는 `Answer:` 패턴만 인식하여 `\boxed{}` 답을 전부 오답 처리함.

**해결:** 커밋 `73fe1fce`에서 위 4개 파일을 복원:
```bash
cd $NEMOTRON_REPO_ROOT
git checkout 73fe1fce -- \
    verl/utils/reward_score/aime.py \
    verl/utils/reward_score/grader.py \
    verl/utils/reward_score/__init__.py \
    verl/workers/reward_manager/naive.py
```

**예방:** VERL repo를 pull/merge할 때 위 4개 파일이 변경되지 않았는지 반드시 확인.

### 2. AIME eval에서 load_format=dummy (모델 weight 미로드)

**증상:** AIME24/25에서 acc=0%, 모든 출력이 무의미한 텍스트

**원인:** VERL `RolloutConfig`의 `load_format` 기본값이 `"dummy"`이며,
`val_only=True` 모드에서는 자동으로 `"auto"`로 전환되지 않음.

**해결:** `eval_scripts/run_eval_aime24.sh`에 명시적으로 추가:
```
actor_rollout_ref.rollout.load_format=auto
```

### 3. MemAgent: method 불일치 (openai vs recurrent)

**증상:** MemAgent 점수가 이전 결과 대비 ~15-20% 낮음

**원인:** `unified_eval.py`의 기본값은 `--memagent_method openai`이지만,
`MemAgent/run_eval.sh` 직접 실행 시 기본값은 `--method recurrent`.
`recurrent` 모드에서는 rope_scaling(YaRN)이 temp dir에 자동 주입되어
long-context 성능이 올라가지만, `openai` 모드에서는 주입되지 않음.

**주의:** rope 주입은 MemAgent 내부에서 temp dir로만 처리되므로
모델의 원본 config.json은 변경되지 않음. 다른 벤치마크(IFEval, AIME, LiveBench 등)는
rope injection 없이 정상 동작해야 하므로 모델 config를 직접 수정하면 안 됨.

**해결:** `unified_eval.py` 경유 시 반드시 method를 명시:
```bash
bash run_eval.sh --model /path --benchmarks memagent --memagent_method recurrent
```

### 4. MemAgent: --memagent_tests hqa로 Squad 누락

**증상:** MemAgent 결과에 HQA(7K, 14K)만 있고 Squad(32K, 64K)가 없음

**원인:** `unified_eval.py`의 기본값이 `--memagent_tests hqa`로, HQA만 실행.
전체 4개 테스트(HQA 7K/14K + Squad 32K/64K)를 돌리려면 `--memagent_tests all` 필요.

| 옵션 | 실행 테스트 |
|------|-----------|
| `--memagent_tests hqa` | HQA 50(7K), HQA 100(14K) |
| `--memagent_tests ruler` | Squad 32K, Squad 64K |
| `--memagent_tests all` | 위 4개 전부 |

**해결:**
```bash
bash run_eval.sh --model /path --benchmarks memagent --memagent_method recurrent --memagent_tests all
```

### 5. Tool Use: vLLM 서버 포트 충돌 (이전 프로세스 미정리)

**증상:** 특정 모델/run에서만 전 카테고리 0.0%, result 파일에 `The model does not exist` 404 에러

**원인:** 이전 run의 vLLM 서버 프로세스가 종료되지 않고 포트를 점유.
새 모델 서버가 시작되지 못하고, 이전 모델이 로드된 서버에 요청이 가서 모델명 불일치 404 발생.

**해결:** `run_tool_use_4x.sh`에 각 run 시작 전 cleanup 로직 추가:
```bash
pkill -f "vllm serve" 2>/dev/null || true
sleep 5
```
**주의:** `pkill -f "bfcl"`은 현재 실행 중인 bfcl 프로세스까지 죽일 수 있으므로 사용하지 말 것.

### 6. Tool Use: dtype 불일치 (float16 vs bfloat16)

**증상:** 이전 결과와 현재 결과에서 Live Simple, Live Parallel 등 일부 카테고리 점수가
~1-6% 차이남 (예: whitened_k512 Simple 81.78→78.68, Parallel 75.00→68.75)

**원인:** `QwenGenericHandler`의 `self.dtype`이 `"float16"`으로 하드코딩되어 있었음.
vLLM serve에 `--dtype float16`이 전달되어, 모델의 native bfloat16 weight가 float16으로
캐스팅됨. 이전(3월 초)에는 `QwenHandler`(기본 `dtype="bfloat16"`)를 사용해서 bfloat16으로
서빙했기 때문에 결과가 달랐음.

temperature=0.001(거의 greedy)이어도 dtype 차이로 logit 값이 미세하게 달라져
argmax가 바뀌는 케이스가 발생함.

**해결:** `qwen_generic.py`에서 환경변수 `BFCL_DTYPE`으로 제어 가능하도록 수정:
```python
self.dtype = os.environ.get("BFCL_DTYPE", "bfloat16")
```

**사용법:**
```bash
# bfloat16 (기본값, 이전 결과와 일치)
SKIP_EVAL_VENV=1 bash run_tool_use_4x.sh

# float16으로 변경
SKIP_EVAL_VENV=1 BFCL_DTYPE=float16 bash run_tool_use_4x.sh
```

**주의:** `qwen_generic.py`가 여러 경로에 복사본이 존재함. 모두 동기화해야 하고,
`__pycache__`가 이전 버전을 캐시할 수 있으므로 수정 후 반드시 삭제:
```bash
find <repo>/Tool_use/ <repo>/Tool_use/ \
    -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
```

해당 파일 위치 (전부 동기화 필요):
- `<repo>/Tool_use/berkeley-function-call-leaderboard/bfcl/model_handler/local_inference/qwen_generic.py`
- `<repo>/Tool_use/berkeley-function-call-leaderboard/bfcl/model_handler/local_inference/qwen_generic.py`
- `<repo>/Tool_use/handlers/qwen_generic.py`
- `<repo>/Tool_use/handlers/qwen_generic.py`

### 7. Tool Use: SKIP_EVAL_VENV로 vLLM 버전 선택

**배경:** 시스템 Python에 vLLM 0.17.0, eval venv에 vLLM 0.11.0이 설치되어 있음.
`Tool_use/env.sh`가 기본적으로 eval venv를 활성화하므로 vLLM 0.11.0이 사용됨.

**증상:** vLLM 0.17.0으로 돌리려 했으나 `EVAL_VENV=""`가 nohup 환경에서 제대로 전달되지 않아
여전히 eval venv(0.11.0)가 활성화됨.

**해결:** `Tool_use/env.sh`에 `SKIP_EVAL_VENV` 환경변수 지원 추가:
```bash
# env.sh 내부 로직:
if [[ "${SKIP_EVAL_VENV:-0}" == "1" ]]; then
    echo "시스템 Python 사용"
elif [[ -f "${EXISTING_VENV}/bin/activate" ]]; then
    source "${EXISTING_VENV}/bin/activate"
fi
```

**사용법:**
```bash
# eval venv (vLLM 0.11.0) — 기본
bash run_tool_use_4x.sh

# 시스템 Python (vLLM 0.17.0)
SKIP_EVAL_VENV=1 bash run_tool_use_4x.sh
```

**주의:** vLLM 0.17.0은 `openai>=1.99.1` 필요. 시스템에 openai 2.30.0 설치 완료.
`SKIP_EVAL_VENV`는 `Tool_use/env.sh`에만 적용되므로 다른 벤치마크(AIME, IFEval 등)에는 영향 없음.

### 8. Tool Use: 프롬프트에 tool schema 누락 + 모델별 prompt mode

**증상:** ToolRL-grpo-cold(tool calling RL 학습 모델)가 base 모델과 거의 같은 점수.
모든 모델이 75~77% 범위에 몰림.

**원인:** `QwenHandler._format_prompt(messages, function)`이 `function` 인자를 완전히 무시.
프롬프트에 tool schema(`<tools>` 블록)가 포함되지 않아, 모델이 어떤 함수를 호출해야 하는지
알 수 없는 상태에서 평가됨. → RL로 학습한 tool calling 능력이 발휘 안 됨.

**해결:** `QwenGenericHandler`에 tool-calling 프롬프트 포맷 추가:
- `BFCL_PROMPT_MODE=tool` (기본): `<tools>` 블록 + `<tool_call>` 응답 포맷 지시
- `BFCL_PROMPT_MODE=plain`: 기존 ChatML (tool 정보 없음)

또한 `decode_ast`/`decode_execute`에 `<tool_call>` XML 파싱 추가:
- 모델 응답이 `<tool_call>{"name": ..., "arguments": ...}</tool_call>` 형태이면 파싱
- `<tool_call>` 없으면 기존 Python 스타일 파서로 fallback

**모델별 prompt mode 설정 (run_tool_use_4x.sh):**

| 모델 | prompt mode | 이유 |
|------|------------|------|
| Qwen2.5-7B-Instruct | `plain` | tool calling RL 미학습 |
| ReasonFlux-Coder-7B | `plain` | 코딩 특화, tool calling RL 미학습 |
| RL-MemoryAgent-7B | `plain` | memory 특화, tool calling RL 미학습 |
| ToolRL-grpo-cold | `tool` | tool calling RL 학습 모델 |
| merged 모델 전체 | `tool` | ToolRL expert 포함 머지 |

**사용법:**
```bash
# 전체 실행 (모델별 자동 설정)
SKIP_EVAL_VENV=1 BFCL_DTYPE=bfloat16 SAVE_DIR=/path/to/save bash run_tool_use_4x.sh

# 수동 override (모든 모델에 동일 적용)
BFCL_PROMPT_MODE=plain bash run_tool_use_4x.sh
```

### 9. Tool Use: openai 패키지 다운그레이드 문제

**증상:** `ModuleNotFoundError: No module named 'openai.types.responses'`로 vLLM 서버 시작 실패

**원인:** 다른 벤치마크(AIME 등)를 eval venv로 실행한 뒤 시스템 Python의 openai 패키지가
1.58.0으로 다운그레이드됨. vLLM 0.17.0은 openai >=1.99.1이 필요.

**해결:** tool_use 실행 전 openai 버전 확인 및 재설치:
```bash
python3 -c "import openai; print(openai.__version__)"
# 2.25.0 미만이면:
pip install "openai>=2.25.0"
```

**예방:** eval venv와 시스템 Python의 패키지가 서로 간섭하지 않도록 주의.
`SKIP_EVAL_VENV=1`로 tool_use를 돌리기 전에 반드시 시스템 openai 버전을 확인할 것.
