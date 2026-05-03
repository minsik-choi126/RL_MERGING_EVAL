"""Compute per-column W from each expert's INPUT activation magnitudes at its
own anti-key positions (bottom-K% by Δlog p), multiplied by the column norm
of the expert's own weight matrix:

    W_col[i, ℓ, c] = ‖W_i,ℓ[:,c]‖₂  ·  mean_{h ∈ neg_i}  | x_expert_i(ℓ, h, c) |

where neg_i is the bottom-K% answer positions of expert i's own prompts,
ranked by Δlog p = log p_expert_i(target_h) − log p_base(target_h)
(positions where expert UNDER-performs base — "anti-key" / failure tokens).

The factor ‖W[:,c]‖₂ corresponds to the symmetric output-contribution proxy
for column c (analogous to row-side which uses |y_r| = |W[r,:]·x| and thus
implicitly contains W).

Δlog p comes pre-computed from the per_query npz (Step 1). Only the expert
is forwarded; we hook the INPUT to each Linear module to collect per-channel
activation magnitudes at the bottom-K positions.

Output:  outputs/W_col_neg_top<pct>_perexpert.npz   (shape (3, d_in) per layer)
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

EXPERT_ORDER = ["ifeval", "math", "coding"]


def load_task(per_query_dir: Path, task: str, expert: str, key_top_frac: float):
    """Return (seqs, prompt_lens, masks, cutoff_max_selected) for the
    bottom-K% positions by Δlog p of `expert`."""
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
    # bottom-K positions (most negative Δlog p)
    bot_idx = np.argpartition(delta, n_key_target - 1)[:n_key_target]
    key_mask = np.zeros(n_total, dtype=np.float32)
    key_mask[bot_idx] = 1.0
    cutoff_max_selected = float(delta[bot_idx].max())

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
          f"cutoff_max_selected={cutoff_max_selected:.6f}  "
          f"neg={n_key}/{n_total} ({n_key/n_total*100:.2f}%)")
    return seqs, pls, masks, cutoff_max_selected


def _hook_factory(state: dict, store: dict):
    def mk(nm: str):
        def hook(_m, inputs, _out):
            x = inputs[0]
            if x is None: return
            a, b = state["a"], state["b"]
            if x.dim() == 3:
                # always slice to answer span [a:b] so per-token mask aligns
                store[nm] = x[0, a:b].detach().clone()
            elif x.dim() == 2:
                store[nm] = x[a:b].detach().clone()
        return hook
    return mk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key_top_frac", type=float, default=0.05,
                    help="per-expert BOTTOM fraction by Δlog p counted as anti-key (default 0.05)")
    ap.add_argument("--in_dir", default=str(DEFAULT_PER_QUERY))
    ap.add_argument("--out", default=None,
                    help="output .npz path; default = outputs/W_col_neg_top<pct>_perexpert.npz")
    ap.add_argument("--ifeval", required=True, help="ifeval RL expert path")
    ap.add_argument("--math",   required=True, help="math RL expert path")
    ap.add_argument("--coding", required=True, help="coding RL expert path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_prompts", type=int, default=None)
    args = ap.parse_args()

    expert_paths = {"ifeval": args.ifeval, "math": args.math, "coding": args.coding}
    in_dir = Path(args.in_dir)
    pct = int(round(args.key_top_frac * 100))
    if args.out is None:
        out_path = ROOT / "outputs" / f"W_col_neg_top{pct:02d}_perexpert.npz"
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    acc_c: dict = {e: {} for e in EXPERT_ORDER}
    n_key_per_expert: dict = {e: 0 for e in EXPERT_ORDER}
    expert_colnorms: dict = {e: {} for e in EXPERT_ORDER}   # ‖W_e[:,c]‖₂ per layer
    linear_names_global: list = []

    for ei, expert_name in enumerate(EXPERT_ORDER):
        seqs, pls, masks, _ = load_task(in_dir, expert_name, expert_name, args.key_top_frac)
        if args.max_prompts:
            seqs = seqs[:args.max_prompts]; pls = pls[:args.max_prompts]; masks = masks[:args.max_prompts]

        print(f"\n========= expert {ei+1}/{len(EXPERT_ORDER)}: {expert_name} =========")
        expert_path = expert_paths[expert_name]
        print(f"[load] expert {expert_name} on {args.device}: {expert_path}")
        expert = AutoModelForCausalLM.from_pretrained(
            _resolve_model_path(expert_path), torch_dtype=torch.bfloat16,
            device_map={"": args.device}).eval()

        # Capture column norms for this expert (per Linear, in fp32)
        for name, mod in expert.named_modules():
            if isinstance(mod, nn.Linear):
                W = mod.weight.data.float()                    # (d_out, d_in)
                colnorm = W.norm(dim=0).cpu().numpy().astype(np.float32)   # (d_in,)
                expert_colnorms[expert_name][f"{name}.weight"] = colnorm

        x_acts: dict = {}
        expert_state = {"a": 0, "b": 0}
        handles = []
        linear_names = []
        mk = _hook_factory(expert_state, x_acts)
        for name, mod in expert.named_modules():
            if isinstance(mod, nn.Linear):
                linear_names.append(name)
                handles.append(mod.register_forward_hook(mk(name)))
        if not linear_names_global:
            linear_names_global = list(linear_names)

        t0 = time.time()
        with torch.no_grad():
            for qi, (seq, pl, km) in enumerate(zip(seqs, pls, masks), 1):
                Lseq = int(seq.shape[0])
                if Lseq <= pl: continue
                x_acts.clear()
                a, b = pl - 1, Lseq - 1
                expert_state["a"] = a; expert_state["b"] = b
                _ = expert(seq.unsqueeze(0).to(args.device), use_cache=False)
                m = km.to(args.device)
                n_key_per_expert[expert_name] += int(m.sum().item())
                for name in linear_names:
                    x = x_acts.get(name)
                    if x is None: continue
                    x_abs = x.float().abs()              # (ans_len, d_in)
                    layer_key = f"{name}.weight"
                    k = f"xabs_neg::{layer_key}"
                    bucket = acc_c[expert_name]
                    if k not in bucket:
                        d_in = x_abs.shape[1]
                        bucket[k] = torch.zeros(d_in, dtype=torch.float32, device=args.device)
                    bucket[k].add_((m[:, None] * x_abs).sum(dim=0))
                    del x, x_abs
                x_acts.clear()
                if qi % 10 == 0 or qi == len(seqs):
                    dt = time.time() - t0
                    eta = dt / qi * (len(seqs) - qi)
                    print(f"  {qi}/{len(seqs)} ({dt:.0f}s, ETA {eta:.0f}s)", flush=True)

        for h in handles: h.remove()
        for k in list(acc_c[expert_name]):
            acc_c[expert_name][k] = acc_c[expert_name][k].cpu()
        del expert, x_acts
        try:
            torch.cuda.empty_cache(); gc.collect()
        except Exception as ex:
            print(f"[warn] empty_cache failed: {ex}", flush=True)

    # ── Save: column W file with ‖W[:,c]‖ factor per expert ─────────────────
    payload = {}
    for name in linear_names_global:
        layer_key = f"{name}.weight"
        stack = []
        for e in EXPERT_ORDER:
            k = f"xabs_neg::{layer_key}"
            if k not in acc_c[e]: continue
            mean_xabs = acc_c[e][k].cpu().numpy() / max(n_key_per_expert[e], 1)
            colnorm = expert_colnorms[e].get(layer_key)
            if colnorm is None or colnorm.shape != mean_xabs.shape:
                continue
            ω = (colnorm * mean_xabs).astype(np.float32)
            stack.append(ω)
        if len(stack) != len(EXPERT_ORDER): continue
        if len({a.shape[0] for a in stack}) != 1: continue
        payload[layer_key] = np.stack(stack, axis=0).astype(np.float32)
    np.savez_compressed(out_path, **payload)
    print(f"\n[save] {out_path}  ({out_path.stat().st_size/1e6:.1f} MB, {len(payload)} layers)")
    print("[done]")


if __name__ == "__main__":
    main()
