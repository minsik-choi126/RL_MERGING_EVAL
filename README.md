# RL_MERGING_EVAL

One-shot evaluation pipeline for AIME 24/25/26, IFEval, LiveBench,
LiveCodeBench, BFCL tool-use, and MemAgent RULER — used for RL / model-merging
experiments. GPU count and model context length are auto-detected; everything
else is toggled via CLI flags.

```
 1. Install env   →  bash setup.sh
 2. Prepare data  →  (done by setup.sh; re-run any time with scripts/download_data.py)
 3. Run eval      →  bash run_eval.sh --model <path> --benchmarks <list> [--tp N ...]
 4. Check results →  ./results/run_NNN/{summary.json, logs/, Coding/, Tool_use/, MemAgent/}
```

---

## 1. Install environment

```bash
git clone https://github.com/minsik-choi126/RL_MERGING_EVAL.git
cd RL_MERGING_EVAL
bash setup.sh
```

`setup.sh` is idempotent. On first run it:

1. Creates a Python venv at `./.eval` (override with `EVAL_VENV=/abs/path`).
2. Installs pip deps, vLLM, flash-attn, `bfcl-eval`.
3. Clones VERL (`volcengine/verl`, shallow) into `./external/verl`.
4. Downloads AIME + MemAgent RULER-HQA data.

Granular skips: `--skip-venv --skip-pip --skip-vllm --skip-bfcl --skip-verl --skip-data`.

Optional: copy `config.env.example` → `config.env` and edit paths
(`EVAL_VENV`, `CACHE_BASE`, `EVAL_DATA_ROOT`, `NEMOTRON_REPO_ROOT`, …).
`config.env` is git-ignored and auto-sourced by every script.

> **AIME note.** VERL upstream rewards `Answer:` only, not `\boxed{}`. If AIME
> reports 0 %, drop the custom reward files (`aime.py`, `grader.py`,
> `reward_score/__init__.py`, `reward_manager/naive.py`) into
> `$NEMOTRON_REPO_ROOT/verl/…`. See `docs/EVAL_GUIDE.md §Known Pitfalls #1`.

## 2. Prepare data and models

Re-run data prep any time (safe to repeat):

```bash
python scripts/download_data.py                       # AIME + MemAgent
python scripts/download_data.py --variants aime24     # just aime24
python scripts/download_data.py --skip-memagent       # AIME only
python scripts/download_data.py --force               # overwrite
```

Artefacts:

| Where                                         | What                                    |
|-----------------------------------------------|-----------------------------------------|
| `$EVAL_DATA_ROOT/aime{24,25,26}/*.parquet`    | VERL-ready AIME parquets                |
| `./MemAgent/hotpotqa/eval_*.json`             | RULER HotpotQA at multiple distances    |
| HuggingFace `google/IFEval` (on-the-fly load) | IFEval                                   |
| BFCL package data (bundled)                   | Tool-use test cases                     |
| `$CODING_DATA_DIR/{LiveBench,LiveCodeBench}.json` | CURE-formatted coding datasets — not auto-downloaded (5 GB combined). Obtain from the CURE eval repo and point `CODING_DATA_DIR` at their directory, or drop/symlink them into `./Coding/data/`. |

**Models.** Pass any local directory with a valid `config.json`. Context length,
`rope_scaling`, and attention-head layout are auto-detected.

## 3. Run benchmarks

Single entry point — toggle benchmarks in the `--benchmarks` list.

```bash
bash run_eval.sh --model <path> --benchmarks <list> [--tp N] [--gpu_per_engine N] ...
```

Benchmarks:

| Flag             | Measures                                          |
|------------------|---------------------------------------------------|
| `aime24`         | AIME 2024 (n=8, temp=0.6)                         |
| `aime25`         | AIME 2025                                          |
| `aime26`         | AIME 2026                                          |
| `ifeval`         | IFEval strict instruction-following                |
| `livebench`      | LiveBench coding                                   |
| `livecodebench`  | LiveCodeBench                                      |
| `tool_use`       | BFCL (live + non-live function calling)            |
| `memagent`       | RULER HQA long-context (7K/14K; add Squad 32K/64K) |
| `coding`         | alias: `livebench + livecodebench`                 |
| `all`            | alias: every benchmark above                       |

Manual knobs (override auto-detection):

| Flag                | Default       | Notes                                                    |
|---------------------|---------------|----------------------------------------------------------|
| `--tp N`            | 4             | Tensor-parallel for IFEval / tool_use / MemAgent         |
| `--n_gpus N`        | auto-detect   | GPUs for AIME (VERL `n_gpus_per_node`)                   |
| `--gpu_per_engine N`| 1             | Coding: GPUs per vLLM engine (use 4 for 7B models)       |
| `--memagent_tests`  | `hqa`         | `hqa` / `ruler` / `all`                                  |
| `--memagent_method` | `openai`      | `openai` / `recurrent` (recurrent injects YaRN rope)     |
| `--temperature X`   | benchmark-def.| Override sampling temperature (coding)                    |
| `--chat_template`   | tokenizer def.| `qwen` / `cure`                                          |
| `--run N`           | auto-increment| Force a specific `run_N` directory                       |
| `--dry_run`         | off           | Print the commands without executing                      |

GPU count is auto-detected via `nvidia-smi --list-gpus` (works on 2×A6000,
4×A100, 8×H100, anywhere). TP constraint: TP must divide both
`num_attention_heads` and `num_key_value_heads` of the model.

### Examples

```bash
# Everything, auto-configured:
bash run_eval.sh --model /path/to/model --benchmarks all

# Two math benches only, TP=2:
bash run_eval.sh --model /path/to/model --benchmarks aime24 aime25 --tp 2

# 7B-sized model, coding with 1 big engine (TP=4):
bash run_eval.sh --model /path/to/Qwen2.5-7B --benchmarks coding --gpu_per_engine 4

# MemAgent full suite with YaRN rope injection:
bash run_eval.sh --model /path/to/model --benchmarks memagent \
    --memagent_method recurrent --memagent_tests all --tp 2

# Loop over many models:
for M in /path/to/A /path/to/B /path/to/C; do
    bash run_eval.sh --model "$M" --benchmarks aime25 ifeval --tp 2
done

# Many models × N repeats for tool_use variance:
N_RUNS=4 bash run_tool_use_4x.sh --models-file models.txt

# First-time user shortcut — run setup inline:
bash run_eval.sh --auto-setup --model /path/to/model --benchmarks all
```

## 4. Check results

Every run creates a fresh auto-incrementing directory:

```
results/
└── run_001/
    ├── eval_config.json              # exact CLI + auto-detected settings
    ├── summary.json                  # per-benchmark status, exit code, elapsed
    ├── logs/<benchmark>.log          # full stdout/stderr per benchmark
    ├── Coding/<model>.txt            # LiveBench + LiveCodeBench (ACC, UT)
    ├── Tool_use/<model>/*.json       # BFCL scores per category
    └── MemAgent/ruler_hqa_<dist>/*.jsonl
```

AIME and IFEval use VERL's conventional layout:
```
$OUTPUT_ROOT/<model>/evaluation_output/{aime24,aime25,aime26,ifeval}/
```
(`OUTPUT_ROOT` defaults to `./results/verl_outputs`.)

Run number is auto-incremented from existing `run_NNN` dirs; override with
`--run N`. Re-running the same benchmark overwrites in place.

## 5. Worked example (A6000 × 4, coding benchmark)

```bash
# 1) env
git clone https://github.com/minsik-choi126/RL_MERGING_EVAL.git
cd RL_MERGING_EVAL
bash setup.sh                          # venv + pip deps + vLLM + VERL + data

# 2) data (coding benches auto-download on first run; nothing extra here)

# 3) eval — 7B model, 4×A6000 = 1 engine × TP4
bash run_eval.sh \
    --model /abs/path/to/Qwen2.5-7B-Instruct \
    --benchmarks coding \
    --gpu_per_engine 4

# (for a 1.7B model on the same 4 GPUs use 4 engines × TP1:)
# bash run_eval.sh --model /abs/path/to/Qwen3-1.7B --benchmarks coding --gpu_per_engine 1

# 4) results
cat results/run_001/Coding/*.txt        # ACC / UT for LiveBench + LiveCodeBench
cat results/run_001/summary.json        # per-benchmark status + elapsed
less results/run_001/logs/livebench.log # full stdout if something looks off
```

---

## Troubleshooting

See `docs/EVAL_GUIDE.md` — past failure modes (NFS busy, AIME 0 % from missing
reward, MemAgent method mismatch, BFCL dtype drift, OOM tuning, etc.).
