"""Build proxy per_query/{task}.npz for Qwen3-1.7B + 3 RL experts.

Calibration set = the **training distributions** of the three RL experts
(NOT eval benches), to keep W computation leak-free w.r.t. eval:

    ifeval  ←  nvidia/Nemotron-Cascade-RL-Instruction-Following
    math    ←  nvidia/Nemotron-Cascade-RL-Math
    lucy  ←  nvidia/Nemotron-RL-lucy-competitive_coding

For each task and query in the sampled subset, we run teacher-forcing through
base + each expert and save:

    base_lp        (T,)              float32  log p_base(target_{h+1} | h)
    expert_lp      (n_experts+1, T)  float32  including base at index 0
    tokens         (T,)              int32    answer tokens
    full_tokens    (full_T,)         int32    prompt + answer tokens
    seq_lens       (n_q,)            int32    answer length per query
    full_seq_lens  (n_q,)            int32    prompt+answer length per query
    prompt_lens    (n_q,)            int32    prompt length per query
    expert_names   (n_experts+1,)    str      ['base', 'ifeval', 'math', 'lucy']
    query_sources  (n_q,)            str      data source label
    query_ids      (n_q,)            int32    sampling index

Output: <out_dir>/{ifeval,math,lucy}.npz   (override via --out_dir)

Usage:
    python prep_proxy_qwen3.py \\
        --base    Qwen/Qwen3-1.7B \\
        --ifeval  /path/to/ifeval_expert  \\
        --math    /path/to/math_expert    \\
        --lucy  /path/to/coding_expert  \\
        --n_queries 128 --seed 0 \\
        --out_dir ../data/per_query
"""
from __future__ import annotations
import argparse
import gc
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# HF dataset ids per RL training recipe (Nemotron-Cascade) — fallback if the
# local jsonl under data/training/ is missing.
HF_DATASETS = {
    "ifeval": "nvidia/Nemotron-Cascade-RL-Instruction-Following",
    "math":   "nvidia/Nemotron-Cascade-RL-Math",
    "lucy": "nvidia/Nemotron-RL-lucy-competitive_coding",
}

# Maximum answer length we consume per query — keeps memory bounded.
# Math/lucy can have very long reasoning; we truncate at this many tokens.
MAX_ANS_TOKENS = 4096
MAX_PROMPT_TOKENS = 8192


# ─── 1. Schema-aware (prompt, answer) extraction ────────────────────────────
def _first_str(*candidates) -> str | None:
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c
    return None


def extract_pa(task: str, row: dict) -> Tuple[str, str] | None:
    """Try a handful of common schemas. Returns (prompt, answer) or None."""
    # Common Nemotron-style: 'messages' = [{role, content}, ...]
    if "messages" in row and isinstance(row["messages"], list):
        msgs = row["messages"]
        # Pick first user msg as prompt, last assistant msg as answer
        prompt_parts, answer_parts = [], []
        for m in msgs:
            if not isinstance(m, dict): continue
            role = m.get("role")
            content = m.get("content", "")
            if role in ("user", "system"):
                prompt_parts.append(content)
            elif role == "assistant":
                answer_parts.append(content)
        prompt = "\n".join(p for p in prompt_parts if p).strip()
        answer = "\n".join(p for p in answer_parts if p).strip()
        if prompt and answer:
            return prompt, answer

    # Flat fields
    prompt = _first_str(
        row.get("prompt"), row.get("question"), row.get("input"),
        row.get("instruction"), row.get("problem"),
    )
    answer = _first_str(
        row.get("solution"), row.get("answer"), row.get("response"),
        row.get("completion"), row.get("output"),
        row.get("ground_truth"), row.get("reference"),
    )
    if prompt and answer:
        return prompt, answer

    return None


# ─── 2. Sampling ────────────────────────────────────────────────────────────
def sample_records(task: str, n: int, seed: int,
                   training_dir: Path,
                   max_attempts: int = 5000) -> List[Tuple[str, str]]:
    """Sample n (prompt, answer) pairs.

    Prefer the local jsonl under training_dir/{task}.jsonl (run
    download_training_data.py once); fall back to streaming HF if missing.
    """
    local = training_dir / f"{task}.jsonl"
    rng = np.random.default_rng(seed)

    if local.exists():
        print(f"[{task}] reading local jsonl: {local}")
        out = []
        import json
        with open(local) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("prompt") and d.get("answer"):
                    out.append((d["prompt"], d["answer"]))
        if len(out) < n:
            raise RuntimeError(
                f"[{task}] local file has only {len(out)} rows < n={n}. "
                f"Run: python download_training_data.py --n {max(n, 256)}"
            )
        idx = rng.permutation(len(out))[:n]
        sampled = [out[int(i)] for i in idx]
        print(f"  → sampled {len(sampled)} from {len(out)} local rows")
        return sampled

    # Fallback: stream HF
    from datasets import load_dataset
    hf_id = HF_DATASETS[task]
    print(f"[{task}] local missing; streaming {hf_id}")
    ds = load_dataset(hf_id, split="train", streaming=True)
    out = []
    seen = 0
    for row in ds:
        seen += 1
        pa = extract_pa(task, row)
        if pa is None:
            if seen <= 3:
                print(f"  [warn] could not extract prompt/answer from row keys={list(row.keys())[:6]}")
            continue
        out.append(pa)
        if len(out) >= max_attempts:
            break
    if len(out) < n:
        raise RuntimeError(
            f"[{task}] only got {len(out)} usable rows from {hf_id} "
            f"(scanned {seen}). Adjust extract_pa() schemas."
        )
    idx = rng.permutation(len(out))[:n]
    sampled = [out[int(i)] for i in idx]
    print(f"  → sampled {len(sampled)} from first {len(out)} usable rows")
    return sampled


# ─── 3. Teacher-forcing + log p extraction ──────────────────────────────────
@torch.no_grad()
def compute_log_probs(
    model_path: str, tokenizer, queries: List[Tuple[str, str]],
    device: str, dtype: torch.dtype = torch.bfloat16,
) -> Tuple[List[np.ndarray], List[int], List[int], List[np.ndarray]]:
    """Returns (log_probs[i], prompt_lens[i], full_lens[i], full_tokens[i]).

    For each query, do teacher forcing on prompt+answer, return per-position
    log p of target_{h+1} at answer positions (length = answer_len).
    """
    print(f"  loading {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map={"": device},
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).eval()

    log_probs_list, prompt_lens, full_lens, full_tokens = [], [], [], []
    for qi, (prompt, answer) in enumerate(queries):
        # Apply chat template if available; else fall back to raw text concat
        try:
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=True,
            )
        except Exception:
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ans_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

        # Truncate if too long
        prompt_ids = prompt_ids[:MAX_PROMPT_TOKENS]
        ans_ids = ans_ids[:MAX_ANS_TOKENS]
        if not ans_ids:
            log_probs_list.append(np.zeros(0, dtype=np.float32))
            prompt_lens.append(len(prompt_ids))
            full_lens.append(len(prompt_ids))
            full_tokens.append(np.array(prompt_ids, dtype=np.int32))
            continue

        full = prompt_ids + ans_ids
        L = len(full)
        pl = len(prompt_ids)
        inp = torch.tensor([full], dtype=torch.long, device=device)
        out = model(inp)
        # logits shape (1, L, V); position h predicts token at h+1
        logits = out.logits[0]  # (L, V)
        # We need log p at answer positions: indices pl-1 .. L-2 (inclusive)
        # to predict tokens at positions pl .. L-1 (the answer tokens)
        ans_slice = logits[pl - 1: L - 1].float()
        log_softmax = torch.log_softmax(ans_slice, dim=-1)
        target_ids = torch.tensor(ans_ids, dtype=torch.long, device=device)
        seq_lp = log_softmax.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        log_probs_list.append(seq_lp.cpu().numpy().astype(np.float32))
        prompt_lens.append(pl)
        full_lens.append(L)
        full_tokens.append(np.array(full, dtype=np.int32))

        if (qi + 1) % 16 == 0 or qi == len(queries) - 1:
            print(f"    {qi+1}/{len(queries)}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return log_probs_list, prompt_lens, full_lens, full_tokens


# ─── 4. Per-task driver ─────────────────────────────────────────────────────
def build_task(
    task: str, base: str, experts: Dict[str, str],
    n: int, seed: int, device: str, out_dir: Path, training_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{task}.npz"
    if out_file.exists():
        print(f"[{task}] {out_file} exists → skip")
        return

    queries = sample_records(task, n=n, seed=seed, training_dir=training_dir)
    sources = [task] * len(queries)

    # Use base's tokenizer for everyone (experts share it)
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run base first (we use its tokens/lengths as canonical)
    print(f"[{task}] running base ({base})")
    base_lps, prompt_lens, full_lens, full_tokens = compute_log_probs(
        base, tokenizer, queries, device,
    )

    # Run each expert; tokens must match base lengths (they share tokenizer)
    expert_order = ["base"] + list(experts.keys())
    expert_lps_per_q: List[List[np.ndarray]] = [[] for _ in queries]
    for qi in range(len(queries)):
        expert_lps_per_q[qi].append(base_lps[qi])

    for ename, epath in experts.items():
        print(f"[{task}] running expert {ename} ({epath})")
        elps, _, _, _ = compute_log_probs(epath, tokenizer, queries, device)
        # Sanity: expert log_probs should match base length per query
        for qi, lp in enumerate(elps):
            if lp.shape != base_lps[qi].shape:
                # safety: pad/truncate
                T = base_lps[qi].shape[0]
                lp2 = np.zeros(T, dtype=np.float32)
                lp2[: min(T, lp.shape[0])] = lp[: min(T, lp.shape[0])]
                lp = lp2
            expert_lps_per_q[qi].append(lp)

    seq_lens = np.array([lp.shape[0] for lp in base_lps], dtype=np.int32)
    full_seq_lens = np.array(full_lens, dtype=np.int32)
    prompt_lens_arr = np.array(prompt_lens, dtype=np.int32)
    flat_full_tokens = np.concatenate(full_tokens).astype(np.int32) if full_tokens else np.zeros(0, dtype=np.int32)
    flat_base_lp = np.concatenate(base_lps).astype(np.float32) if base_lps else np.zeros(0, dtype=np.float32)
    expert_lp_stacked = np.stack(
        [np.concatenate([elps[ei] for elps in expert_lps_per_q]).astype(np.float32)
         for ei in range(len(expert_order))],
        axis=0,
    )
    # tokens (answer-only) = full_tokens after each prompt
    tokens_list = []
    for ft, pl, sl in zip(full_tokens, prompt_lens, seq_lens):
        tokens_list.append(ft[pl: pl + sl])
    flat_tokens = np.concatenate(tokens_list).astype(np.int32) if tokens_list else np.zeros(0, dtype=np.int32)

    np.savez_compressed(
        out_file,
        base_lp=flat_base_lp,
        expert_lp=expert_lp_stacked,
        tokens=flat_tokens,
        full_tokens=flat_full_tokens,
        seq_lens=seq_lens,
        full_seq_lens=full_seq_lens,
        prompt_lens=prompt_lens_arr,
        expert_names=np.array(expert_order, dtype=object),
        query_sources=np.array(sources, dtype=object),
        query_ids=np.arange(len(queries), dtype=np.int32),
    )
    print(f"[{task}] saved → {out_file}")


# ─── 5. Main ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B",
                    help="Base model path or HuggingFace id")
    ap.add_argument("--ifeval", required=True, help="ifeval RL expert path")
    ap.add_argument("--math",   required=True, help="math RL expert path")
    ap.add_argument("--lucy", required=True, help="lucy RL expert path")
    ap.add_argument("--n_queries", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42,
                    help="default 42; matches the download_training_data.py seed")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir",
                    default=str(Path(__file__).resolve().parent.parent / "data" / "per_query"),
                    help="output dir for per_query/*.npz")
    ap.add_argument("--training_dir",
                    default=str(Path(__file__).resolve().parent.parent / "data" / "training"),
                    help="dir with local {task}.jsonl files (run download_training_data.py first)")
    ap.add_argument("--tasks", nargs="+", default=["ifeval", "math", "lucy"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    training_dir = Path(args.training_dir)
    expert_paths = {"ifeval": args.ifeval, "math": args.math, "lucy": args.lucy}

    for task in args.tasks:
        if task not in HF_DATASETS:
            raise ValueError(f"unknown task {task}; choose from {list(HF_DATASETS)}")
        build_task(
            task=task,
            base=args.base,
            experts=expert_paths,
            n=args.n_queries,
            seed=args.seed,
            device=args.device,
            out_dir=out_dir,
            training_dir=training_dir,
        )


if __name__ == "__main__":
    main()
