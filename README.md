# RL_MERGING_EVAL

모델 머지 실험에서 사용하는 **통합 평가 파이프라인**입니다. 한 번의 명령으로
AIME / IFEval / LiveBench / LiveCodeBench / BFCL Tool-Use / MemAgent(RULER) 을
같은 환경에서 재현 가능하게 돌릴 수 있도록 wrapper 와 환경변수만 정리한 구조입니다.

평가 로직 자체는 각 벤치마크의 공식/업스트림 코드(VERL, vLLM, BFCL, RULER 등)를
그대로 사용합니다. 이 레포는 그 위에 **오케스트레이션 · 캐시 경로 · GPU 수 자동
감지 · 결과 수집** 만 얹는 얇은 레이어입니다.

---

## 0. 빠른 실행

```bash
# 한번에 전부
bash run_eval.sh --model /path/to/model --benchmarks all

# 특정 벤치만
bash run_eval.sh --model /path/to/model --benchmarks aime24 aime25 ifeval
bash run_eval.sh --model /path/to/model --benchmarks coding          # livebench + livecodebench
bash run_eval.sh --model /path/to/model --benchmarks tool_use
bash run_eval.sh --model /path/to/model --benchmarks memagent --memagent_method recurrent --memagent_tests all

# 명령어만 보고 싶을 때
bash run_eval.sh --model /path/to/model --benchmarks all --dry_run
```

모든 옵션은 `unified_eval.py --help` 참고. GPU 수는 `nvidia-smi` 로 자동 감지되고
Tensor-Parallel / engine 배치도 자동으로 계산됩니다 (A6000 2장이든 A100 4장이든
코드 변경 없이 동작).

---

## 1. 지원 벤치마크

| Benchmark       | Alias         | What It Measures                         | Framework              |
|-----------------|---------------|------------------------------------------|------------------------|
| `aime24`        |               | AIME 2024 (30문항, pass@1 × n=8)        | VERL (`verl.trainer`)  |
| `aime25`        |               | AIME 2025                                 | VERL                   |
| `aime26`        |               | AIME 2026                                 | VERL                   |
| `ifeval`        |               | IFEval (strict instruction-following)    | vLLM + 자체 scorer     |
| `livebench`     | `coding`      | LiveBench coding                         | vLLM multi-engine      |
| `livecodebench` | `coding`      | LiveCodeBench                             | vLLM multi-engine      |
| `tool_use`      |               | BFCL live / non-live function calling    | BFCL CLI + vLLM        |
| `memagent`      |               | RULER HotpotQA long-context              | vLLM serve + OpenAI API|

특수 alias:
- `all` — 위 전체
- `coding` — `livebench` + `livecodebench`

---

## 2. 환경 (현재 테스트된 조합)

| Component       | 버전                                        |
|-----------------|---------------------------------------------|
| GPU             | A6000 × 2 / A100 × 4 / H100 × 4 (자동 감지) |
| CUDA            | 12.1+                                       |
| Python          | 3.10 / 3.11                                 |
| PyTorch         | 2.4+                                        |
| vLLM            | 0.7.0 ~ 0.11.0 (Tool_use 는 0.17.0 가능)    |
| VERL            | nemotron-merge 브랜치 (AIME reward 포함)    |
| flash-attn      | 2.7+                                        |

이 레포는 위 런타임을 _이미_ 가지고 있다는 전제 하에 동작합니다. 설치 자체는
벤치마크별로 다르고 이미지/플랫폼에 따라 변하므로 여기에 고정하지 않습니다.
필요 라이브러리 예시는 [`requirements-eval.txt`](./requirements-eval.txt) 를
참고하세요.

### GPU 수 자동 감지

- `unified_eval.py::detect_gpu_count()` 가 `nvidia-smi --list-gpus` 로 자동 탐지
- AIME: 감지된 GPU 수가 그대로 `n_gpus_per_node` 에 들어감
- Coding: 감지된 GPU 수 / `gpu_per_engine` 로 vLLM 엔진 그룹 자동 구성
- Tool_use / IFEval / MemAgent: `--tp` 가 감지된 GPU 수를 초과하지 않도록 체크

```bash
# 2×A6000 환경 예시 — 자동 감지 결과 그대로 실행
bash run_eval.sh --model /path/to/Qwen3-1.7B --benchmarks all

# 4×A100 + 7B 모델 — coding 만 TP=4 한 엔진으로 돌리고 싶을 때
bash run_eval.sh --model /path/to/Qwen2.5-7B --benchmarks coding --gpu_per_engine 4
```

TP 제약 (tensor parallelism):
- Qwen3-1.7B (heads=16, kv_heads=8) → TP ∈ {1, 2, 4, 8}
- Qwen2.5-7B (heads=28, kv_heads=4) → TP ∈ {1, 2, 4}

TP 와 GPU 수가 호환 안 되면 자동으로 `--tp 1` 로 떨어지지 않으니
사용자가 명시해야 합니다 (`--tp 2` 등).

---

## 3. 초기 세팅

### 3-1. 클론 & 디렉토리 구조

```bash
git clone https://github.com/minsik-choi126/RL_MERGING_EVAL.git
cd RL_MERGING_EVAL
```

```
RL_MERGING_EVAL/
├── run_eval.sh              # 모든 벤치마크 진입점
├── unified_eval.py          # 오케스트레이터 (모델 auto-detect + dispatch)
├── run_tool_use_4x.sh       # 여러 모델 × N회 반복 Tool_use 실행
├── models.txt.example       # run_tool_use_4x.sh 용 모델 리스트 예시
├── eval_scripts/            # AIME24/25/26 + IFEval 래퍼 + 공통 env
│   ├── common_eval_env.sh   # ★ 캐시·venv·PYTHONPATH 모든 환경변수
│   ├── run_eval_aime24.sh   # VERL main_ppo 호출
│   ├── run_eval_aime25.sh
│   ├── run_eval_aime26.sh
│   ├── run_eval_ifeval.sh   # vLLM + 자체 IFEval scorer
│   ├── ifeval_eval.py
│   ├── collect_results.py   # 결과 수집 유틸
│   └── collect_ifeval_results.py
├── Coding/                  # LiveBench / LiveCodeBench
│   ├── env.sh               # → common_eval_env.sh
│   ├── run_eval.sh
│   └── evaluation/
│       ├── eval.py          # CURE 기반 평가 스크립트
│       └── evaluation_config.py
├── Tool_use/                # BFCL
│   ├── env.sh               # → common_eval_env.sh + PYTHONPATH
│   ├── run_eval.sh          # bfcl generate / evaluate
│   ├── handlers/            # 로컬 inference handler (qwen_generic)
│   ├── print_bfcl_table.py
│   └── summarize_results.py
├── MemAgent/                # RULER HQA / Squad long-context
│   ├── env.sh               # → common_eval_env.sh + DATAROOT
│   ├── run_eval.sh
│   ├── serve/llm070.py      # vLLM 0.7 경로 serve entrypoint
│   └── taskutils/memory_eval/
│       ├── run.py           # 동적 Config + HQA / RULER runner
│       ├── ruler_hqa.py
│       ├── ruler_general.py
│       └── utils/           # openai / recurrent / boxed APIs
├── docs/EVAL_GUIDE.md       # 상세 가이드 + 과거 삽질 기록
├── config.env.example       # 로컬 환경변수 템플릿
└── requirements-eval.txt
```

### 3-2. `config.env` 로 경로만 지정

`config.env.example` 를 `config.env` 로 복사한 뒤 본인 환경에 맞춰 수정하세요.
git 에는 커밋되지 않습니다 (`.gitignore`).

```bash
cp config.env.example config.env
$EDITOR config.env
# 이후
set -a; source config.env; set +a
bash run_eval.sh --model /path/to/model --benchmarks all
```

주요 변수:

| 변수                  | 기본값                                       | 설명                                              |
|-----------------------|----------------------------------------------|---------------------------------------------------|
| `EVAL_VENV`           | `$REPO_ROOT/.eval`                           | 사용할 venv 경로. 없으면 시스템 Python fallback  |
| `SKIP_EVAL_VENV`      | `0`                                          | `1` 이면 venv 활성화를 완전히 건너뜀             |
| `CACHE_BASE`          | `$HOME/.cache/rl_merging_eval`               | HF / vLLM / Triton / FlashInfer 캐시 전부 여기    |
| `EVAL_DATA_ROOT`      | `$REPO_ROOT/data`                            | AIME / IFEval parquet 들이 있는 루트             |
| `NEMOTRON_REPO_ROOT`  | `$REPO_ROOT/external/verl_nemotron_merge`    | 커스텀 AIME reward 가 있는 VERL 체크아웃          |
| `TMPDIR`, `RAY_TMPDIR`| `/tmp/eval_tmp_$USER`, `/tmp/ray_$USER`      | 반드시 LOCAL disk (NFS 쓰면 .nfs silly-rename)    |
| `OUTPUT_ROOT`         | `$REPO_ROOT/results/verl_outputs`            | AIME / IFEval 의 VERL 출력 루트                  |

### 3-3. 데이터 (`EVAL_DATA_ROOT`) 준비

AIME 24/25/26 와 IFEval 은 아래와 같은 layout 의 parquet 들을 읽습니다:

```
$EVAL_DATA_ROOT/
├── aime24/test_verl_ready_with_instruction.parquet
├── aime25/test_verl_ready_with_instruction.parquet
├── aime26/test_verl_ready_with_instruction.parquet
└── ifeval/ ...           # IFEval 은 HuggingFace `google/IFEval` 에서 로드하므로 로컬 불필요
```

MemAgent 의 HotpotQA 원본 데이터(`eval_*.json`, ~8 GB)는 레포에 포함하지
않습니다. RULER 업스트림의 공식 다운로드 스크립트 또는 HuggingFace
(`RULER-benchmarks/*`) 미러에서 받은 뒤 `MemAgent/hotpotqa/` 아래에 놓으세요.

### 3-4. VERL (AIME reward)

AIME 는 VERL 의 `verl.trainer.main_ppo` 를 `val_only=True` 로 호출합니다.
기본 VERL 업스트림은 `Answer:` 패턴만 인식해서 `\boxed{}` 답을 전부
오답 처리하므로, **AIME 전용 reward 를 담은 커스텀 체크아웃이 필요합니다**.
`$NEMOTRON_REPO_ROOT` 를 아래 4개 파일이 있는 VERL clone 에 맞춰 주세요:

- `verl/utils/reward_score/aime.py`
- `verl/utils/reward_score/grader.py`
- `verl/utils/reward_score/__init__.py`
- `verl/workers/reward_manager/naive.py`

상세는 `docs/EVAL_GUIDE.md` 의 _"Known Pitfalls #1"_ 참고.

### 3-5. BFCL (Tool_use)

`Tool_use/run_eval.sh` 는 `bfcl generate` / `bfcl evaluate` CLI 를 호출합니다.
두 가지 설치 옵션:

1. **pip 로 설치** (가장 간단)
    ```bash
    pip install bfcl-eval
    ```
    `Tool_use/handlers/qwen_generic.py` 는 `<tools>` 블록 / dtype 제어 / `<tool_call>`
    XML 파싱이 패치된 버전입니다. BFCL 의 `model_handler/local_inference/` 경로에
    복사해 넣으면 됩니다.

2. **BFCL 레포를 `Tool_use/berkeley-function-call-leaderboard/` 에 clone**
    → `Tool_use/env.sh` 가 이 경로를 자동으로 `PYTHONPATH` 앞에 얹습니다.

모델별 prompt mode (plain / tool) 는 `run_tool_use_4x.sh` 의
`--models-file models.txt` 에서 지정합니다. `models.txt.example` 참고.

---

## 4. 벤치마크별 상세

### AIME 24 / 25 / 26

```bash
bash run_eval.sh --model $M --benchmarks aime24 aime25 aime26
```

- VERL `main_ppo` 를 `val_only=True` 로 실행
- `n=8` 샘플링 (pass@1 × 8 repeats, temperature=0.6, top_p=0.95)
- `max_response_length` 는 모델 `config.json` 의 `max_position_embeddings` 를
  보고 `min(30720, max_pos - 2048)` 로 자동 계산. 오버라이드하려면
  `MAX_RESPONSE_LENGTH=8192 bash run_eval.sh ...`
- 결과: `$OUTPUT_ROOT/<model>/evaluation_output/aime24/` (VERL 기본 output)

### IFEval

```bash
bash run_eval.sh --model $M --benchmarks ifeval --tp 2
```

- HF `google/IFEval` 로드 → vLLM 로 greedy 샘플링 → 공식 규칙 기반 scorer
- 메트릭: `prompt_level_strict_acc`, `inst_level_strict_acc`
- 결과: `$OUTPUT_ROOT/<model>/evaluation_output/ifeval/{summary.json, results.jsonl}`

### LiveBench / LiveCodeBench

```bash
# 7B 모델 + 4 GPU → 1 engine × TP4
bash run_eval.sh --model $M --benchmarks coding --gpu_per_engine 4

# 1.7B + 4 GPU → 4 engines × TP1 (병렬 증가)
bash run_eval.sh --model $M --benchmarks coding --gpu_per_engine 1

# 2 GPU 환경 (예: A6000 2장)
bash run_eval.sh --model $M --benchmarks coding --gpu_per_engine 1   # 2 engines × TP1
bash run_eval.sh --model $M --benchmarks coding --gpu_per_engine 2   # 1 engine  × TP2
```

GPU 레이아웃은 자동 계산 (`total_gpus / gpu_per_engine`). `nvidia-smi` 로
자동 감지되는 `TOTAL_GPUS` 는 `TOTAL_GPUS=8 bash run_eval.sh ...` 로 오버라이드
가능합니다.

### Tool_use (BFCL)

```bash
# 단일 모델
bash run_eval.sh --model $M --benchmarks tool_use --tp 2

# 여러 모델 + 반복 (variance estimation)
bash run_tool_use_4x.sh --models-file models.txt
N_RUNS=4 bash run_tool_use_4x.sh --models-file models.txt
```

- `bfcl generate` → vLLM 백엔드로 live + non_live 카테고리 생성
- `bfcl evaluate` → 각 카테고리 score JSON
- dtype 고정: `BFCL_DTYPE=bfloat16` (기본) / `float16`
- Prompt mode: `BFCL_PROMPT_MODE=tool|plain`
  (tool-calling RL 학습 안 한 모델은 `plain` 이 스코어가 높음)

### MemAgent (RULER)

```bash
# HQA 만 (2개 테스트: 7K, 14K)
bash run_eval.sh --model $M --benchmarks memagent --memagent_tests hqa

# HQA + Squad 전체 (4개: HQA 7K/14K, Squad 32K/64K)
bash run_eval.sh --model $M --benchmarks memagent \
    --memagent_method recurrent --memagent_tests all

# TP 조정 (7B 이상)
bash run_eval.sh --model $M --benchmarks memagent --tp 2 --memagent_method recurrent
```

`--memagent_method recurrent` 를 쓰면 rope_scaling(YaRN) 이 임시 모델 디렉토리에
자동 주입되어 long-context 정확도가 올라갑니다. 원본 모델 `config.json` 은
변경되지 않으므로 다른 벤치마크에는 영향 없습니다.

---

## 5. 결과 위치

`run_NNN` 은 자동으로 증가합니다 (`--run N` 으로 명시 가능).

```
results/
└── run_001/
    ├── eval_config.json       # 이번 실행 설정 (모델 경로, TP, 벤치 목록 등)
    ├── summary.json           # 벤치별 status · 소요시간
    ├── logs/                  # 벤치별 stdout/stderr
    ├── Coding/<model>.txt     # LiveBench + LiveCodeBench 결과
    ├── Tool_use/<model>/      # BFCL score JSON 들
    └── MemAgent/ruler_hqa_*/  # HQA / Squad 결과 JSONL
```

AIME 와 IFEval 은 VERL 관례에 따라 `$OUTPUT_ROOT/<model>/evaluation_output/`
아래에 저장됩니다 (위 `OUTPUT_ROOT` 환경변수).

---

## 6. 트러블슈팅

상세 사례는 [`docs/EVAL_GUIDE.md`](./docs/EVAL_GUIDE.md) 의 _"Known Pitfalls"_
섹션 참고. 대표적인 것만 요약:

- **AIME acc=0%** → VERL 의 `aime.py` / `grader.py` / reward `__init__.py` /
  `naive.py` 4개 파일이 upstream merge 로 덮어씌워진 상태. 커스텀 체크아웃 복원.
- **AIME 무의미한 텍스트** → `actor_rollout_ref.rollout.load_format=auto` 빠진
  상태. 이 레포의 스크립트엔 이미 들어가 있음.
- **MemAgent 점수 하락** → `--memagent_method recurrent` 대신 `openai` 로 돌린
  경우. YaRN rope-scaling 이 주입 안 됨.
- **MemAgent 테스트가 2개만** → 기본값 `--memagent_tests hqa` 사용. Squad 까지
  원하면 `--memagent_tests all`.
- **Tool_use 전 카테고리 0%** → 직전 run 의 `vllm serve` 가 포트를 잡고 있음.
  `run_tool_use_4x.sh` 는 각 run 전에 `pkill -f "vllm serve"` 를 수행.
- **Tool_use 점수 미묘하게 다름** → dtype 불일치. `BFCL_DTYPE=bfloat16` 고정.
- **NFS `Device or resource busy`** → `TMPDIR` / `RAY_TMPDIR` 가 NFS 위에 있음.
  `/tmp` 같은 LOCAL disk 로 바꾸고 `ray stop --force; rm -rf /tmp/ray_* /tmp/eval_tmp_*`.

---

## 7. 여러 모델 루프 예시

```bash
for model in /path/to/Model-A /path/to/Model-B /path/to/Model-C; do
    bash run_eval.sh --model "$model" --benchmarks aime25 aime26 ifeval --tp 2
done
```

또는 Tool_use 만 N 회 반복:

```bash
N_RUNS=4 bash run_tool_use_4x.sh --models-file models.txt
```

결과는 `results/tool_use_4x/<model>/run_{1..N}/` 아래에 쌓이고, 마지막에
`summary.json` 으로 run 평균이 저장됩니다.

---

## 8. License / 출처

- 이 레포의 wrapper 스크립트와 오케스트레이터(`unified_eval.py`, `run_*.sh`)는
  내부 실험 재현 용도로 작성된 코드입니다.
- 평가 로직 자체는 업스트림:
  - AIME / IFEval 파이프라인: [VERL](https://github.com/volcengine/verl)
  - LiveBench / LiveCodeBench 러너: CURE eval
  - BFCL: [Gorilla BFCL](https://github.com/ShishirPatil/gorilla)
  - MemAgent RULER: [ByteDance-Seed/MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent) (Apache-2.0)
  - vLLM / flash-attn / Ray 등
- 각 업스트림의 라이선스/규정을 따르세요.
