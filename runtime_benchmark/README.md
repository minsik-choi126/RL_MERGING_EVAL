# Runtime Benchmark — Ours vs TSV

Self-contained folder for measuring end-to-end **merge runtime** of two
3-expert merge methods on Qwen2.5-7B-Instruct + 3 RL-finetuned experts:

| method | what's measured | GPU usage |
|---|---|---|
| **Ours** (KT-col-neg + polar + renorm) | **W extraction** (Phase A) + **Merge** (Phase B) | A = DP=N (configurable),  B = 1 GPU |
| **TSV** (Task Singular Vectors, CVPR 2025) | **Merge** only | 1 GPU |

For fair comparison, the **merge** step always runs on a single GPU
(`cuda:0`). The only step that scales with N GPUs is W extraction (which has
no analogue in TSV).

## Reference numbers (2× A6000)

These were measured during the experiment that produced this folder; your
machine will differ but the ratios should be similar.

| method | W extract | merge | total |
|---|---|---|---|
| TSV | – | 27m07s | 27m07s |
| **Ours (DP=2)** | 11m34s | 33m29s | **45m03s** |

Other baselines (provided for reference, not in this folder): TIES (chunked GPU) ~14m52s; RAM 5m07s; RAM+ 6m50s.

## Folder layout

```
runtime_benchmark/
├── README.md                   # (this file)
├── data/per_query_rltrain/     # 13 MB — per-query RL probe data (input to extract_w.py)
│   ├── coding.npz              # 128 prompts (coding domain)
│   ├── tool.npz                # 128 prompts (tool-use domain)
│   └── memory.npz              # 128 prompts (long-context memory; ~25k tokens avg)
├── helpers.py                  # shared model I/O (load/save state_dict)
├── extract_w.py                # W_col extraction with DP=N
├── merge_ours.py               # ktcol_polar_renorm merge (single GPU, optimized)
├── merge_tsv.py                # TSV merge (single GPU, baseline)
├── run_ours.sh                 # Ours wrapper: Phase A (DP=N) + Phase B + timing
├── run_tsv.sh                  # TSV wrapper: merge + timing
└── download_models.sh          # HuggingFace downloader for the 4 models
```

## Models

You need 1 base + 3 RL-finetuned experts (~15 GB each fp32, ~7 GB bf16 saved).

| role     | env var          | default HF id |
|----------|------------------|---------------|
| base     | `BASE_MODEL`     | `Qwen/Qwen2.5-7B-Instruct` |
| coding   | `EXPERT_CODING`  | `Gen-Verse/ReasonFlux-Coder-7B` |
| tool     | `EXPERT_TOOL`    | `emrgnt-cmplxty/Qwen2.5-7B-Instruct-ToolRL-grpo-cold` |
| memory   | `EXPERT_MEMORY`  | `BytedTsinghua-SIA/RL-MemoryAgent-7B` |

If you have local copies, point the env vars at the directories. If the
HF id resolves in your local HF cache, the scripts use that automatically.

## Pre-flight

```bash
# Python deps (any recent CUDA-enabled torch will do)
pip install torch numpy tqdm transformers safetensors huggingface_hub

# One-time: pull models (~60 GB total)
bash download_models.sh
```

If you already have these models elsewhere (e.g. on a shared NFS), skip the
download and override env vars instead:

```bash
export BASE_MODEL=/some/path/Qwen2.5-7B-Instruct
export EXPERT_CODING=/some/path/ReasonFlux-Coder-7B
# ...
```

## Run

### Ours

```bash
bash run_ours.sh                  # default DP=2  (uses cuda:0 + cuda:1 for Phase A)
bash run_ours.sh --dp 4           # 4-GPU machine: 4-way prompt sharding for Phase A
bash run_ours.sh --dp 1           # single-GPU run (slowest, but apples-to-apples vs TSV)
```

Outputs (in this folder):

- `W_col_neg_top10.npz`      — extracted W matrix
- `merged_ours/`             — final merged HF checkpoint (safetensors)
- `logs/phaseA_W.log`, `logs/phaseB_merge.log`

End-of-run summary printed to stdout, e.g.:

```
════════════════════════════════════════════════════════════════
  Ours runtime SUMMARY   (DP=2)
════════════════════════════════════════════════════════════════
  phase                                wall(s)  wall(mm:ss)
  Phase A (W extract, DP=2)             694.49  11:34
  Phase B (merge, 1 GPU)               2008.74  33:29
  TOTAL                                2703.23  45:03
════════════════════════════════════════════════════════════════
```

### TSV

```bash
bash run_tsv.sh
```

Outputs:

- `merged_tsv/`                     — final merged HF checkpoint
- `logs/tsv.log`

## Algorithm notes

### Ours — W extraction (`extract_w.py`)

For each expert e ∈ {coding_rl, tool_rl, memory_rl}:

1. Load per-query records from `data/per_query_rltrain/<task>.npz`.
   Each record holds the full prompt+answer tokens, log-probs of the answer
   under both the base model and this expert, and the answer length.
2. Compute Δlogp = expert_lp − base_lp per answer-token. The **bottom-K%**
   tokens by Δlogp (K=10% by default) are the "key tokens" — those most
   suppressed by the expert relative to base.
3. Forward each prompt through the expert (bf16, no_grad, no_cache); a
   fused forward hook on every `nn.Linear` accumulates
       Σ_t mask[t] · |x_ℓ(t)[c]|
   per (layer, input-channel). The accumulator lives directly in GPU memory
   — no `.detach().clone()` per call.
4. Divide by total `n_key` for that expert.

DP=N data parallelism: each expert's prompt set is sharded into N chunks,
one per GPU. Workers are `mp.Process(spawn=True)`; partial accumulators are
summed at the end.

### Ours — Merge (`merge_ours.py`)

Per 2D layer ℓ, with τ_i = W_expert_i − W_base and per-expert column
weighting D_c (from `--w_col_file`):

```
Y_i        = τ_i · D_c^{1/2}                    # weighted (col-only)
Y_i        = U_y · diag(σ_y) · V_y^T            # full SVD
K_i        = smallest k s.t. Σ σ_y[:k]² ≥ 0.90 · Σ σ_y²
τ_i^(K)    = (U_y[:,:K] · diag(σ_y[:K]) · V_y[:K]^T) · D_c^{-1/2}

# Original-space orthonormal bases (needed for cross-expert polar):
M_right_i  = V_y[:,:K]^T · D_c^{-1/2}            # (K × d_in)
QR(M_right_i^T) → Q_r_i (d_in × K), R_r_i (K × K)
middle_i   = diag(σ_y[:K]) · R_r_i^T             # (K × K) — small
SVD(middle_i) → U_m_i, S_m_i, V_m_i
U_k_i      = U_y[:,:K] · U_m_i                   # (d_out × K) orthonormal
V_k_i      = Q_r_i · V_m_i^T                     # (d_in × K)  orthonormal
S_k_i      = S_m_i

# Polar alignment + per-expert renorm:
U_hat        = polar(cat(U_k_i)),  V_hat = polar(cat(V_k_i))
τ_aligned_i  = U_hat[blocks_i] · diag(S_k_i) · V_hat[blocks_i]^T
α_i          = ‖τ_i‖ / ‖τ_aligned_i‖
merged_τ_ℓ   = Σ_i α_i · τ_aligned_i
```

The QR + small-K×K SVD replaces what would otherwise be a second full-rank
SVD of (d_out × d_in). For `lm_head` (152064 × 3584 with K~1k) that drops
the dominant cost from ≈2 ×10¹² ops → ≈1.5 ×10¹¹ ops per expert per layer.

### TSV — Merge (`merge_tsv.py`)

Per 2D layer:

```
SVD(τ_i) → U_i, S_i, V_i^T per expert i
sum_u  = [U_1[:,:k] | U_2[:,:k] | U_3[:,:k]]      # (d_out × N·k)
sum_v  = [V_1[:k,:] ; V_2[:k,:] ; V_3[:k,:]]      # (N·k × d_in)
sum_s  = concat(S_1[:k] | S_2[:k] | S_3[:k])
u_u, _, v_u = SVD(sum_u);  u_v, _, v_v = SVD(sum_v)
merged_τ = (u_u·v_u) · diag(sum_s) · (u_v·v_v)
```

with k = ⌊d_min/N⌋ by default. 1D layers: rolling mean.

## Troubleshooting

- **OOM during Phase A on `memory_rl`**: memory prompts average ~25k
  tokens, ~28 k max. With bf16 + SDPA on a 48 GB GPU it should fit at
  batch=1; lower-VRAM GPUs may need to drop `--max_prompts` or skip the
  expert.
- **`tensor_parallel` not used**: it forces eager attention which OOMs on
  long memory prompts. We use **data parallel (prompt sharding)** instead.
- **Different number of params per expert**: the helpers use safetensors-
  level `state_dict` keys; if your HF copies have an extra wrapper key
  prefix (e.g. `model.`), edit `helpers.load_state_dict` accordingly.
- **TSV OOM**: TSV merge is dominated by per-layer SVD; should fit on a
  single 48 GB GPU. If your card is smaller, reduce energy/k by passing
  `--k 256` or similar to `merge_tsv.py`.
- **TIES baseline**: not in this folder. The original `run_ties` does
  `vstack(3 × flat_fp32_7B)` (84 GB) which OOMs on a single GPU. A chunked
  variant is documented elsewhere.

## What is *not* benchmarked here

- **Eval time** (LiveBench / LiveCodeBench / BFCL / MemAgent) — these are
  separate tasks; the scripts here only time **merging**.
- **TIES / RAM / RAM+ baselines** — those merge codes live in the parent
  project (`merge_baseline.py`) and aren't bundled to keep this folder
  small.
