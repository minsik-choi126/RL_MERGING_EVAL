# KT-Merge — Qwen3-1.7B 3-Way RL Expert Merging Pipeline

End-to-end recipe for **base + 3 RL experts → merged models**, with our
KT-merging method (column-wise weighting on negative-Δlogp positions) **and**
9 baselines, ready for a 6-bench evaluation.

> Base: `Qwen/Qwen3-1.7B`
> Experts: ifeval-RL, math-RL, coding-RL  (Nemotron-Cascade recipes)
> Eval: ifeval / aime24 / aime25 / aime26 / livebench / livecodebench

---

## 0. Quick start

### Full pipeline (ours + 9 baselines)

```bash
cd <path-to>/KT_merge

# Install deps (Python 3.10 + CUDA-matched torch)
pip install -r requirements.txt

# Place expert models under models/{ifeval,math,coding}/  (or symlinks)

export CUDA_VISIBLE_DEVICES=0

# Full run: download → targets → proxy → W_col → merge × (1 ours + 9 baselines)
nohup bash run_pipeline.sh > outputs/pipeline.log 2>&1 &
tail -f outputs/pipeline.log
```

### Run only **ours** (skip baselines)

```bash
SKIP_BASELINES=1 bash run_pipeline.sh
```

→ Builds proxy + W_col + the **ours** merge only.
Output: `outputs/merges/W_col_neg_top<pct>/model.safetensors`.

### Skip already-built artifacts

```bash
# Re-compute W + re-merge ours only (proxy already built)
SKIP_DOWNLOAD=1 SKIP_GEN_TARGETS=1 SKIP_PREP=1 SKIP_BASELINES=1 \
    bash run_pipeline.sh

# Build proxy + W only, no merge
SKIP_MERGE=1 bash run_pipeline.sh
```

### Toggles

| env var | default | meaning |
|---|---|---|
| `KEY_TOP_FRAC` | **0.05** | bottom-K fraction by Δlogp (anti-key tokens) |
| `ENERGY` | 0.90 | KT-Truncation energy threshold |
| `BASE_MODEL` | `Qwen/Qwen3-1.7B` | HF id of base |
| `N_QUERIES` | 128 | proxy queries per task |
| `SEED` | 42 | sampling seed |
| `DEVICE` | `cuda:0` | logical device for every step |
| `SKIP_DOWNLOAD` | 0 | skip Step 0a |
| `SKIP_GEN_TARGETS` | 0 | skip Step 0b |
| `SKIP_PREP` | 0 | skip Step 1 |
| `SKIP_COMPUTE_W` | 0 | skip Step 2 |
| `SKIP_MERGE` | 0 | skip Step 3 |
| `SKIP_BASELINES` | 0 | run **only ours**; skip 9 baselines in Step 3 |

```bash
# Different threshold
KEY_TOP_FRAC=0.10 SKIP_BASELINES=1 bash run_pipeline.sh
KEY_TOP_FRAC=0.20 SKIP_BASELINES=1 bash run_pipeline.sh
```

---

## 1. Pipeline overview

```
data/training/{ifeval,math,coding}_raw.jsonl ← Step 0a (download HF prompts)
            │
            ▼
data/training/{ifeval,math,coding}.jsonl     ← Step 0b (expert-generated targets)
            │
            ▼
data/per_query/{ifeval,math,coding}.npz      ← Step 1 (teacher-force base+experts)
            │
            ▼
outputs/W_col_neg_top<pct>_perexpert.npz     ← Step 2 (col-side W on bottom-K%)
            │
            ▼
outputs/merges/<method>/                     ← Step 3 (ours + 9 baselines)
            │
            ▼
external eval pipeline                       ← Step 4 (6 benches per merge)
```

---

## 2. Method (ours)

For each Linear layer ℓ and expert i, compute a column-side weight from
**bottom-K% Δlogp positions** (anti-key — where expert UNDER-performs base):

```
ω_col[i, ℓ, c] = ‖W_i,ℓ[:, c]‖₂  ·  mean_{h ∈ neg_i} | x_expert_i(ℓ, h, c) |
```

Then apply column-weighted SVD truncation per expert:

```
Y_i,ℓ   = ΔW_i,ℓ · diag(ω_col[i, ℓ])^{1/2}
        SVD(Y) → keep top-K singular values reaching energy ≥ 0.9
τ̂_i,ℓ  = Y^(K)_i,ℓ · diag(ω_col[i, ℓ])^{−1/2}
```

Cross-expert merge with polar alignment + per-expert renormalization
(unchanged from the original KT-Polar). Variant tag: **`ktcol_polar_renorm`**.

---

## 3. Step-by-step

### Step 0 — Build training proxy data

**Stage 0a** ([`scripts/download_training_data.py`](scripts/download_training_data.py)):
streams 3 HuggingFace datasets, samples 128 prompts each (seed 42), saves
`{prompt, answer|null}` JSONL to `data/training/{task}_raw.jsonl`.

| task | HF id |
|---|---|
| ifeval | `nvidia/Nemotron-Cascade-RL-Instruction-Following` |
| math | `nvidia/Nemotron-Cascade-RL-Math` |
| coding | `nvidia/Nemotron-RL-coding-competitive_coding` |

**Stage 0b** ([`scripts/generate_targets.py`](scripts/generate_targets.py)):
fills missing answers via each task's expert (greedy, max_new_tokens=512).

### Step 1 — Build proxy per_query npz ([`scripts/prep_proxy_qwen3.py`](scripts/prep_proxy_qwen3.py))

For each task and `(prompt, answer)`, run teacher-forcing on **base + 3 experts**
and save log-probs at the answer positions. Used downstream to identify
anti-key tokens (bottom-K% by Δlog p).

### Step 2 — Compute W_col ([`scripts/compute_W_col.py`](scripts/compute_W_col.py))

For each expert i (forwarded on its own task), pick the **bottom-K%** answer
positions ranked by Δlog p = log p_expert_i − log p_base (anti-key set).
Hook the **input** to each Linear, accumulate `|z|`, average, then multiply
by per-expert `‖W_i,ℓ[:, c]‖₂`.

```bash
python scripts/compute_W_col.py \
    --key_top_frac 0.05 \
    --ifeval models/ifeval --math models/math --coding models/coding \
    --in_dir data/per_query \
    --out outputs/W_col_neg_top05_perexpert.npz \
    --device cuda:0
```

Output: per-layer 2D tensor of shape `(N=3, d_in)` indexed by
`[ifeval, math, coding]`.

### Step 3 — Merge ([`scripts/merge_baselines.sh`](scripts/merge_baselines.sh))

| method | merge command |
|---|---|
| **ours: W_col_neg_top<pct>** | `merge_ablation.py --variants ktcol_polar_renorm --w_col_file <W>` |
| task_arithmetic | `merge.py --method task_arithmetic` |
| ties / dare_ta | `merge.py --method ties|dare ...` |
| star / tsv | `merge.py --method <name>` |
| iso_c / iso_cts | `merge.py --method iso_c|iso_cts` |
| ram / ram_plus | `merge.py --method ram|ram_plus` |

**Run only ours** (no baselines):
```bash
SKIP_BASELINES=1 bash scripts/merge_baselines.sh
```

Or via the pipeline orchestrator:
```bash
SKIP_BASELINES=1 bash run_pipeline.sh
```

Output: `outputs/merges/W_col_neg_top<pct>/model.safetensors` (symlink into
the auto-detected ablation subdir).

### Step 4 — Evaluate (external)

Plug each `outputs/merges/<method>/` into your own eval pipeline. The benches:
ifeval, aime24, aime25, aime26, livebench, livecodebench.

---

## 4. Variants supported

`scripts/merge_ablation.py` ships **13 variants** for ablation:

| tag | truncate | polar | renorm | kt_mode | notes |
|---|---|---|---|---|---|
| naive | none | — | — | — | TA sum baseline |
| svd / svd_polar / svd_polar_renorm | std SVD | progressively | progressively | — | |
| kt / kt_polar / kt_polar_renorm | KT | progressively | progressively | row | row-side weighting |
| ktcol / ktcol_polar / **ktcol_polar_renorm** | KT | progressively | progressively | col | **OURS** (column-side) |
| kt2s / kt2s_polar / kt2s_polar_renorm | KT | progressively | progressively | 2s | row × col simultaneous |

---

## 5. GPU usage

```bash
export CUDA_VISIBLE_DEVICES=0          # 1 GPU is sufficient
export CUDA_VISIBLE_DEVICES=0,1        # Step 1 / merge_baselines benefit from 2
```

`DEVICE` is the logical device (default `cuda:0`).
Step 2 only forwards each expert (Δlog p is read from per_query npz), single GPU enough.

---

## 6. Files

```
KT_merge/
├── README.md                          ← this file
├── requirements.txt
├── run_pipeline.sh                    ← one-shot orchestrator
├── models/                            ← place expert dirs (or symlinks)
│   ├── ifeval/  math/  coding/
├── data/
│   ├── training/{task}_raw.jsonl      ← Stage 0a
│   ├── training/{task}.jsonl          ← Stage 0b
│   └── per_query/{task}.npz           ← Step 1
├── outputs/
│   ├── W_col_neg_top<pct>_perexpert.npz                     ← Step 2
│   ├── merges/<method>/                                     ← Step 3
│   └── merge_logs/                                          ← per-method logs
├── scripts/
│   ├── download_training_data.py    Stage 0a
│   ├── generate_targets.py          Stage 0b
│   ├── prep_proxy_qwen3.py          Step 1
│   ├── compute_W_col.py             Step 2 (NEW: col-side, neg-K%, with ‖W‖)
│   ├── merge_ktpolar.py             Step 3 helpers (kttrunc_per_expert with W_col)
│   ├── merge_ablation.py            Step 3 entry for ours (13 variants)
│   ├── merge.py                     Step 3 baselines
│   └── merge_baselines.sh           Step 3 driver (SKIP_BASELINES=1 to skip baselines)
└── deps/
    └── kt_merge_helpers.py
```

---

## 7. Notes

- **Idempotent**: every step checks for existing artifacts and skips.
  Resume after a crash by simply re-running `bash run_pipeline.sh`.
- **Disk**: each merged 1.7B model ≈ 3.4 GB float32 / 1.7 GB bf16.
  10 merges (1 ours + 9 baselines) ≈ 20–35 GB. Plus per_query npz (~tens of MB).
- **Save dir validation in `merge.py`**: by default disabled. To restrict
  the output root, `export KT_SAVE_ROOT=/abs/path/you/want` before running.
- **Evaluation**: not part of this pipeline. After Step 3, plug
  `outputs/merges/<method>/` into your own eval pipeline.
