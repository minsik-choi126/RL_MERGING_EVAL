"""Compute per-row W from each expert's activation magnitudes at its own key
positions on the proxy training data.

For each (expert i, layer ℓ, row r):

    W_expert[i, ℓ, r] = mean_{h ∈ key_i}  |y_expert_i(ℓ, h, r)|

where key_i is the top-K% answer positions (default 20%) of expert i's own
prompts, ranked by Δlog p = log p_expert_i(target_h) − log p_base(target_h).
Δlog p comes pre-computed from the per_query npz (Step 1), so the base model
is NOT re-loaded here. Top-k selection is exact via argpartition (no
threshold ties). Only the expert is forwarded; we accumulate |y_expert| at
key positions to build a per-layer (N=3, d_out) tensor indexed by
[ifeval, math, lucy].

Output:  <out>  (default: outputs/W_expert_top<pct>_perexpert.npz)
"""
from __future__ import annotations
import argparse, gc, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "deps"))
from kt_merge_helpers import _resolve_model_path  # type: ignore

ROOT = HERE.parent
DEFAULT_PER_QUERY = ROOT / "data" / "per_query"

# (expert_name, task_name) — task name == .npz filename in per_query
EXPERT_ORDER = ["ifeval", "math", "lucy"]


def load_task(per_query_dir: Path, task: str, expert: str, key_top_frac: float):
    """Load proxy npz for `task`, mark top-K% answer positions of `expert`'s
    own Δlog p as key, return per-prompt sequences/masks."""
    if not (0.0 < key_top_frac <= 1.0):
        raise ValueError(f"key_top_frac must be in (0, 1]; got {key_top_frac}")
    z = np.load(per_query_dir / f"{task}.npz", allow_pickle=True)
    full_tokens = z["full_tokens"]
    full_seq_lens = [int(x) for x in z["full_seq_lens"]]
    prompt_lens = [int(x) for x in z["prompt_lens"]]
    ans_lens = [int(x) for x in z["seq_lens"]]
    en = [str(n) for n in z["expert_names"]]
    if expert not in en:
        raise KeyError(f"{task}.npz expert_names={en} missing {expert!r}")
    base_lp = z["base_lp"].astype(np.float32)
    exp_lp = z["expert_lp"].astype(np.float32)[en.index(expert)]
    delta = exp_lp - base_lp
    n_total = len(delta)
    n_key_target = int(np.ceil(key_top_frac * n_total))
    top_idx = np.argpartition(-delta, n_key_target - 1)[:n_key_target]
    key_mask = np.zeros(n_total, dtype=np.float32)
    key_mask[top_idx] = 1.0
    cutoff_min_selected = float(delta[top_idx].min())

    seqs, pls, masks = [], [], []
    off_full, off_ans = 0, 0
    for pl, flen, alen in zip(prompt_lens, full_seq_lens, ans_lens):
        flen, alen = int(flen), int(alen)
        seq = torch.from_numpy(full_tokens[off_full: off_full + flen].astype(np.int64))
        m = torch.from_numpy(key_mask[off_ans: off_ans + alen].copy())
        seqs.append(seq); pls.append(pl); masks.append(m)
        off_full += flen; off_ans += alen
    n_key = int(key_mask.sum())
    print(f"[load] {expert} ({task}): {len(seqs)} prompts, "
          f"cutoff_min_selected={cutoff_min_selected:.6f}  "
          f"key={n_key}/{n_total} ({n_key/n_total*100:.2f}%)")
    return seqs, pls, masks, cutoff_min_selected


def _hook_factory(state: dict, store: dict):
    def mk(nm: str):
        def hook(_m, _i, output):
            if nm == "lm_head":
                store[nm] = output[0].detach().clone()
            else:
                a, b = state["a"], state["b"]
                store[nm] = output[0, a:b].detach().clone()
        return hook
    return mk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key_top_frac", type=float, default=0.20,
                    help="per-expert top fraction by Δlog p counted as key (default 0.20)")
    ap.add_argument("--in_dir", default=str(DEFAULT_PER_QUERY),
                    help="directory containing {ifeval,math,lucy}.npz")
    ap.add_argument("--out", default=None,
                    help="output .npz path; default = outputs/W_expert_top<pct>_perexpert.npz")
    ap.add_argument("--ifeval", required=True, help="ifeval RL expert path")
    ap.add_argument("--math",   required=True, help="math RL expert path")
    ap.add_argument("--lucy", required=True, help="lucy RL expert path")
    ap.add_argument("--device", default="cuda:0",
                    help="GPU for the expert forward (default cuda:0)")
    ap.add_argument("--max_prompts", type=int, default=None,
                    help="limit prompts per expert (for quick smoke tests)")
    args = ap.parse_args()

    expert_paths = {"ifeval": args.ifeval, "math": args.math, "lucy": args.lucy}
    in_dir = Path(args.in_dir)
    pct = int(round(args.key_top_frac * 100))
    if args.out is None:
        out_path = ROOT / "outputs" / f"W_expert_top{pct:02d}_perexpert.npz"
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-expert accumulators (only key-side; non-key not needed for W_expert)
    acc_e: dict = {e: {} for e in EXPERT_ORDER}
    n_key_per_expert: dict = {e: 0 for e in EXPERT_ORDER}
    linear_names_global: list = []

    cutoffs: dict = {}
    for ei, expert_name in enumerate(EXPERT_ORDER):
        seqs, pls, masks, cutoff = load_task(in_dir, expert_name, expert_name, args.key_top_frac)
        cutoffs[expert_name] = cutoff
        if args.max_prompts:
            seqs = seqs[:args.max_prompts]; pls = pls[:args.max_prompts]; masks = masks[:args.max_prompts]

        print(f"\n========= expert {ei+1}/{len(EXPERT_ORDER)}: {expert_name} =========")
        expert_path = expert_paths[expert_name]
        print(f"[load] expert {expert_name} on {args.device}: {expert_path}")
        expert = AutoModelForCausalLM.from_pretrained(
            _resolve_model_path(expert_path), torch_dtype=torch.bfloat16,
            device_map={"": args.device}).eval()
        expert_acts: dict = {}
        expert_state = {"a": 0, "b": 0}
        expert_handles = []
        linear_names = []
        ek_mk = _hook_factory(expert_state, expert_acts)
        for name, mod in expert.named_modules():
            if isinstance(mod, nn.Linear):
                linear_names.append(name)
                expert_handles.append(mod.register_forward_hook(ek_mk(name)))
        if not linear_names_global:
            linear_names_global = list(linear_names)
        elif linear_names_global != linear_names:
            print(f"[warn] expert {expert_name} has different Linear layout; "
                  f"using current expert's layout for accumulation")
            linear_names_global = list(linear_names)

        t0 = time.time()
        with torch.no_grad():
            for qi, (seq, pl, km) in enumerate(zip(seqs, pls, masks), 1):
                Lseq = int(seq.shape[0])
                if Lseq <= pl:
                    continue
                expert_acts.clear()
                a, b = pl - 1, Lseq - 1
                expert_state["a"] = a; expert_state["b"] = b
                ans_idx = torch.arange(a, b, device=args.device)
                _ = expert(seq.unsqueeze(0).to(args.device),
                            use_cache=False, logits_to_keep=ans_idx)
                m = km.to(args.device)
                n_key_per_expert[expert_name] += int(m.sum().item())
                for name in linear_names:
                    ye = expert_acts[name].float()
                    ye_abs = ye.abs()
                    layer_key = f"{name}.weight"
                    bucket = acc_e[expert_name]
                    k = f"yabs_key::{layer_key}"
                    if k not in bucket:
                        bucket[k] = torch.zeros(ye.shape[1], dtype=torch.float32,
                                                  device=args.device)
                    bucket[k].add_((m[:, None] * ye_abs).sum(dim=0))
                    del ye, ye_abs
                expert_acts.clear()
                if qi % 10 == 0 or qi == len(seqs):
                    dt = time.time() - t0
                    eta = dt / qi * (len(seqs) - qi)
                    print(f"  {qi}/{len(seqs)} ({dt:.0f}s, ETA {eta:.0f}s)", flush=True)

        for h in expert_handles:
            h.remove()
        # CPU off accumulators ASAP so a subsequent CUDA error doesn't lose progress.
        for k in list(acc_e[expert_name]):
            acc_e[expert_name][k] = acc_e[expert_name][k].cpu()
        del expert, expert_acts
        try:
            torch.cuda.empty_cache(); gc.collect()
        except Exception as ex:
            print(f"[warn] empty_cache failed (ignored): {ex}", flush=True)

    # ── Save W file (per-expert 2D per layer) ───────────────────────────────
    payload = {}
    for name in linear_names_global:
        layer_key = f"{name}.weight"
        stack = []
        d_out = None
        for e in EXPERT_ORDER:
            k = f"yabs_key::{layer_key}"
            if k not in acc_e[e]:
                continue
            arr = acc_e[e][k].cpu().numpy() / max(n_key_per_expert[e], 1)
            stack.append(arr)
            d_out = arr.shape[0]
        if not stack or d_out is None:
            continue
        payload[layer_key] = np.stack(stack, axis=0).astype(np.float32)
    np.savez_compressed(out_path, **payload)
    print(f"\n[save] {out_path}  ({out_path.stat().st_size/1e6:.1f} MB, {len(payload)} layers)")
    print("[done]")


if __name__ == "__main__":
    main()
