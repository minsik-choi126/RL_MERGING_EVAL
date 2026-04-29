# KT-Merge — Qwen3-1.7B 3-Way RL Expert Merging Pipeline

End-to-end recipe for **base + 3 RL experts → merged models**, with our
KT-merging method **and** 12 baselines, ready for a 6-bench evaluation.

> Base: `Qwen/Qwen3-1.7B`
> Experts: ifeval-RL, math-RL, coding-RL  (Nemotron-Cascade recipes)
> Eval: ifeval / aime24 / aime25 / aime26 / livebench / livecodebench

---

## 0. Quick start (one-shot)

```bash
cd <path-to>/KT_merge

# 0. Install deps (Python 3.10 + CUDA-matched torch — see requirements.txt)
pip install -r requirements.txt

# 1. Place expert models under models/{ifeval,math,coding}/
#    Each must contain *.safetensors + config.json + tokenizer.* (or be a symlink).

# 2. Pick GPU(s)
export CUDA_VISIBLE_DEVICES=0           # single GPU is enough (default)
                                        # CUDA_VISIBLE_DEVICES=0,1 only speeds up the optional 2-GPU prep_proxy path

# 3. Full run: download → targets → proxy → W_expert → merge × (1 ours + 12 baselines)
nohup bash run_pipeline.sh > outputs/pipeline.log 2>&1 &
tail -f outputs/pipeline.log
```

Step toggles (skip what's already done):

```bash
SKIP_DOWNLOAD=1 SKIP_PREP=1 bash run_pipeline.sh   # only re-compute W + re-merge
SKIP_MERGE=1                bash run_pipeline.sh   # build proxy + W only
```

Hyperparameter overrides:

```bash
KEY_TOP_FRAC=0.20 N_QUERIES=128 \
    bash run_pipeline.sh
```

---

## 1. Pipeline overview

```
data/training/{ifeval,math,coding}_raw.jsonl ← Step 0a (download)
            │
            ▼
data/training/{ifeval,math,coding}.jsonl     ← Step 0b (expert targets)
            │
            ▼
data/per_query/{ifeval,math,coding}.npz      ← Step 1 (teacher-force base+experts)
            │
            ▼
outputs/W_expert_top<pct>_perexpert.npz      ← Step 2 (per-expert top-K% W)
            │
            ▼
outputs/merges/<method>/                     ← Step 3 (13 methods × 1 merge each)
            │
            ▼
external eval pipeline                       ← Step 4 (6 benches per merge)
```

---

## 2. Step-by-step

### Step 0 — Build training proxy data (two stages)

**Stage 0a** ([`scripts/download_training_data.py`](scripts/download_training_data.py)):
streams 3 HuggingFace datasets, samples 128 prompts each (seed 42), saves
`{prompt, answer|null}` JSONL to `data/training/{task}_raw.jsonl`. Math has
gold answers; ifeval/coding RL data ships prompts only.

| task | HF id |
|---|---|
| ifeval | `nvidia/Nemotron-Cascade-RL-Instruction-Following` |
| math | `nvidia/Nemotron-Cascade-RL-Math` |
| coding | `nvidia/Nemotron-RL-coding-competitive_coding` |

**Stage 0b** ([`scripts/generate_targets.py`](scripts/generate_targets.py)):
fills in missing answers by running each task's own expert (greedy, max_new_tokens=512)
on every prompt. Writes the completed `data/training/{task}.jsonl`.

```bash
python scripts/download_training_data.py --n 128 --seed 42
python scripts/generate_targets.py \
    --ifeval models/ifeval --math models/math --coding models/coding \
    --tokenizer_src Qwen/Qwen3-1.7B --device cuda:0
```

### Step 1 — Build proxy per_query npz ([`scripts/prep_proxy_qwen3.py`](scripts/prep_proxy_qwen3.py))

For each task and each sampled `(prompt, answer)`, run teacher-forcing on
**base + 3 experts** and stack log-probs at the answer positions:

```
base_lp        (T,)        log p_base(target_{h+1}|h)
expert_lp      (4, T)      ['base', 'ifeval', 'math', 'coding'] log p
tokens / full_tokens / seq_lens / prompt_lens / ...
```

```bash
python scripts/prep_proxy_qwen3.py \
    --base Qwen/Qwen3-1.7B \
    --ifeval models/ifeval --math models/math --coding models/coding \
    --n_queries 128 --seed 42 --device cuda:0
```

> **Why training data, not eval data?** W's per-row weight encodes "where
> the experts lift target log-prob" on the proxy set. If proxy = eval set,
> we leak eval signal into the merge. Using each expert's RL training
> distribution is leak-free.

### Step 2 — Compute W_expert ([`scripts/compute_W_expert.py`](scripts/compute_W_expert.py))

Δlog p comes pre-computed from Step 1's per_query npz, so the base model
is **not** re-loaded here. For each expert i (forwarded on its own task
only), pick the top-K% answer positions ranked by
Δlog p = log p_expert_i − log p_base (exact top-k via `argpartition`;
default 20%) as the key set, then accumulate

```
W_expert[i, ℓ, r] = mean_{h ∈ key_i}  |y_expert_i(ℓ, h, r)|
```

producing a per-layer 2D tensor of shape `(N=3, d_out)` indexed by
`[ifeval, math, coding]`.

```bash
python scripts/compute_W_expert.py \
    --key_top_frac 0.20 \
    --ifeval models/ifeval --math models/math --coding models/coding \
    --in_dir data/per_query \
    --out outputs/W_expert_top20_perexpert.npz \
    --device cuda:0
```

### Step 3 — Merge (ours + baselines) ([`scripts/merge_baselines.sh`](scripts/merge_baselines.sh))

Runs 1 ours + 12 baselines = **13 merges total**, idempotent skip:

| method | merge command |
|---|---|
| **ours: W_expert_top<pct>** | `merge_ablation.py --variants kt_polar_renorm --w_file <W>` |
| task_arithmetic (1/N mean) | `merge.py --method task_arithmetic` |
| ties / dare_ta / dare_ties | `merge.py --method ties|dare ...` |
| star / cart / tsv | `merge.py --method <name> --device cuda:0` |
| fisher | `merge.py --method fisher` |
| iso_c / iso_cts | `merge.py --method iso_c|iso_cts --device cuda:0` |
| ram / ram_plus | `merge.py --method ram|ram_plus --device cuda:0` |

Output: `outputs/merges/<method>/model.safetensors` per method. The default
ours directory is `outputs/merges/W_expert_top20/` (symlink into the
auto-detected ablation subdir).

### Step 4 — Evaluate (external)

Evaluation is **not part of this pipeline**. After Step 3, plug each
`outputs/merges/<method>/` into your own eval pipeline. The benches we used:
ifeval, aime24, aime25, aime26, livebench, livecodebench.

---

## 3. GPU usage

```bash
export CUDA_VISIBLE_DEVICES=0          # 1 GPU is sufficient (default)
export CUDA_VISIBLE_DEVICES=0,1        # only Step 1 / merge baselines benefit
```

`DEVICE` is the logical device passed to every step (default `cuda:0`).
Step 2 only forwards each expert (base activations are not needed — Δlog p
is read from the per_query npz), so a single GPU is enough.

---

## 4. Files

```
KT_merge/
├── README.md                          ← this file
├── requirements.txt                   ← pip deps
├── run_pipeline.sh                    ← one-shot orchestrator
├── models/                            ← place expert dirs (or symlinks) here
│   ├── ifeval/                        (*.safetensors + config.json + tokenizer.*)
│   ├── math/
│   └── coding/
├── data/
│   ├── training/{task}_raw.jsonl      ← Stage 0a output
│   ├── training/{task}.jsonl          ← Stage 0b output (answers filled)
│   └── per_query/{task}.npz           ← Step 1 output
├── outputs/
│   ├── W_expert_top<pct>_perexpert.npz                   ← Step 2 output
│   ├── merges/<method>/                                  ← Step 3 output (13 dirs)
│   └── merge_logs/                                       ← per-method logs
├── scripts/
│   ├── download_training_data.py    Stage 0a
│   ├── generate_targets.py          Stage 0b
│   ├── prep_proxy_qwen3.py          Step 1
│   ├── compute_W_expert.py          Step 2
│   ├── merge_ktpolar.py             Step 3 helpers
│   ├── merge_ablation.py            Step 3 entry for ours
│   ├── merge.py                     Step 3 baselines
│   └── merge_baselines.sh           Step 3 driver
└── deps/
    └── kt_merge_helpers.py          5 shared helpers
```

---

## 5. Hyperparameters

| flag | default | meaning |
|---|---|---|
| `BASE_MODEL` | `Qwen/Qwen3-1.7B` | HF id, auto-downloaded |
| `N_QUERIES` | 128 | proxy queries per task |
| `SEED` | 42 | sampling seed |
| `KEY_TOP_FRAC` | **0.20** | per-expert top-K fraction by Δlog p (exact top-k) |
| `ENERGY` | 0.90 | KT-Truncation energy threshold |
| `DEVICE` | `cuda:0` | logical device for every step |

---

## 6. Notes

- **Idempotent**: every step checks for existing artifacts and skips.
  Resume after a crash by simply re-running `bash run_pipeline.sh`.
- **Disk**: each merged 1.7B model ≈ 3.4 GB float32 / 1.7 GB bf16.
  13 merges ≈ 25–45 GB. Plus per_query npz (~tens of MB).
- **Save dir validation in `merge.py`**: by default disabled. To restrict
  the output root, `export KT_SAVE_ROOT=/abs/path/you/want` before running.
- **Evaluation**: not part of this pipeline. After Step 3, plug
  `outputs/merges/<method>/` into your own eval pipeline.
