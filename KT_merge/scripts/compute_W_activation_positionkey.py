"""Per-layer row importance W_l using ONLY per-position key events.

Key definition — per position, NOT per vocab token:
    Δ_E(h) = log p_E(target_{h+1} | h) - log p_base(target_{h+1} | h)
    α(h)   = 1 if max_E Δ_E(h) > θ else 0      (E = {coding_rl, tool_rl, memory_rl})

Non-key positions contribute zero to W_l. After accumulation, per-layer
additive-ε + normalize is applied for numerical safety of KT-Truncation:

    ε_l         = ε_scale · median(W_raw_l)
    W_safe_l[r] = (W_raw_l[r] + ε_l) / (median(W_raw_l) + ε_l)

Output: results/ktpolar/W_activation_positionkey.npz
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "deps"))
from kt_merge_helpers import _resolve_model_path  # type: ignore

# KT_merge layout: scripts/, deps/, data/, outputs/
ROOT = _HERE.parent
IN_DIR = ROOT / "data" / "per_query"
OUT_FILE = ROOT / "outputs" / "W_activation_positionkey.npz"

# Defaults overridable via CLI (--base / --tasks / --experts)
BASE = "Qwen/Qwen3-1.7B"
TASKS = ["ifeval", "math", "coding"]
RL_EXPERT_NAMES = ["ifeval", "math", "coding"]   # match per_query/{task}.npz expert_names


def build_per_expert_sequences_and_masks(tasks, threshold: float,
                                            expert_names_per_task: list[str]):
    """Per-expert variant: each task's prompts are masked using ONLY that task's
    own expert's Δ. Each prompt is tagged with an owner_idx (0..N-1) telling
    downstream code which expert/task accumulator to add to.

    expert_names_per_task: list aligned with tasks; the expert whose mask
    should be used for that task's prompts (typically RL_EXPERT_NAMES).

    Returns:
        seqs        : list[Tensor]
        prompt_lens : list[int]
        masks       : list[Tensor]   per-prompt (answer_len,)
        owner_idx   : list[int]      0..len(tasks)-1, which task this prompt belongs to
    """
    assert len(tasks) == len(expert_names_per_task)
    seqs, prompt_lens, masks, owner_idx = [], [], [], []
    n_per_task = []                                 # debug
    for ti, (task, exp_name) in enumerate(zip(tasks, expert_names_per_task)):
        z = np.load(IN_DIR / f"{task}.npz", allow_pickle=True)
        full_tokens = z["full_tokens"]; full_seq_lens = z["full_seq_lens"]
        pls = [int(x) for x in z["prompt_lens"]]
        ans_lens = [int(x) for x in z["seq_lens"]]
        base_lp = z["base_lp"].astype(np.float32)
        exp_lp  = z["expert_lp"].astype(np.float32)

        en_arr = z.get("expert_names")
        if en_arr is None:
            raise KeyError(f"{task}.npz missing expert_names")
        en_list = [str(n) for n in en_arr]
        if exp_name not in en_list:
            raise ValueError(f"expert {exp_name} not in {task}.npz expert_names {en_list}")
        e_col = en_list.index(exp_name)

        delta_e = exp_lp[e_col] - base_lp                                # (N_ans,)
        pos_key = (delta_e > threshold).astype(np.float32)                # per-position

        n_key_task, n_pos_task = 0, 0
        off_full, off_ans = 0, 0
        for pl, flen, alen in zip(pls, full_seq_lens, ans_lens):
            flen, alen = int(flen), int(alen)
            seq = torch.from_numpy(full_tokens[off_full: off_full + flen].astype(np.int64))
            mask = torch.from_numpy(pos_key[off_ans: off_ans + alen].copy())
            seqs.append(seq); prompt_lens.append(pl); masks.append(mask)
            owner_idx.append(ti)
            n_key_task += int(mask.sum().item()); n_pos_task += alen
            off_full += flen; off_ans += alen
        n_per_task.append((task, exp_name, n_key_task, n_pos_task))

    print(f"[per-expert mask] using own-expert Δ>{threshold} on each task's own prompts:")
    for task, exp_name, n_key, n_pos in n_per_task:
        print(f"  {task} ({exp_name}): {n_key}/{n_pos} key positions "
              f"({n_key/n_pos*100:.2f}%)")
    return seqs, prompt_lens, masks, owner_idx


def build_sequences_and_masks(tasks, threshold: float, expert_names_subset=None):
    """Return (seqs, prompt_lens, masks) aligned 1-to-1.

    masks[i]: torch.float32 tensor of shape (answer_len,), 1 at key positions, else 0.

    expert_names_subset: list of RL expert NAMES to use for the Δ mask
        (default = ['coding_rl','tool_rl','memory_rl']). Per-task npz's
        expert_names array is consulted to find the matching column index;
        works regardless of whether 'base' is included in the npz expert_lp.
    """
    if expert_names_subset is None:
        expert_names_subset = list(RL_EXPERT_NAMES)
    seqs, prompt_lens, masks = [], [], []
    n_key_total, n_pos_total = 0, 0
    for task in tasks:
        z = np.load(IN_DIR / f"{task}.npz", allow_pickle=True)
        full_tokens = z["full_tokens"]; full_seq_lens = z["full_seq_lens"]
        pls = [int(x) for x in z["prompt_lens"]]
        ans_lens = [int(x) for x in z["seq_lens"]]

        base_lp = z["base_lp"].astype(np.float32)           # (N_ans,)
        exp_lp  = z["expert_lp"].astype(np.float32)         # (E, N_ans)

        # Resolve subset → column indices via per-task expert_names
        en_arr = z.get("expert_names")
        if en_arr is None:
            raise KeyError(
                f"{task}.npz missing 'expert_names'; cannot resolve experts. "
                f"Re-run token_analysis with the patched script that saves names.")
        en_list = [str(n) for n in en_arr]
        try:
            sel_idx = [en_list.index(n) for n in expert_names_subset]
        except ValueError as e:
            raise ValueError(
                f"expert {e} not in {task}.npz's expert_names {en_list}; "
                f"requested subset = {expert_names_subset}")

        delta   = exp_lp[sel_idx] - base_lp[None, :]        # (n_experts_sel, N_ans)
        pos_key = (delta.max(axis=0) > threshold).astype(np.float32)  # (N_ans,)

        off_full, off_ans = 0, 0
        for pl, flen, alen in zip(pls, full_seq_lens, ans_lens):
            flen, alen = int(flen), int(alen)
            seq = torch.from_numpy(full_tokens[off_full: off_full + flen].astype(np.int64))
            mask = torch.from_numpy(pos_key[off_ans: off_ans + alen].copy())
            seqs.append(seq); prompt_lens.append(pl); masks.append(mask)
            n_key_total += int(mask.sum().item()); n_pos_total += alen
            off_full += flen; off_ans += alen

        # Sanity: per-task position-level stat
        task_key = int(pos_key.sum())
        print(f"[load] {task}: {task_key}/{len(pos_key)} key positions "
              f"({task_key/len(pos_key)*100:.2f}%)")

    print(f"[load] overall Δ>{threshold}: {n_key_total}/{n_pos_total} "
          f"({n_key_total/n_pos_total*100:.2f}%)")
    return seqs, prompt_lens, masks


def main():
    global IN_DIR   # must come BEFORE any reference to IN_DIR (incl. argparse default)
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Δ threshold (in nats) for key position event")
    ap.add_argument("--eps_scale", type=float, default=0.01,
                    help="ε = eps_scale · median(W_raw); then W_safe = (W_raw + ε) / (median + ε)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(OUT_FILE))
    ap.add_argument("--in_dir", default=str(IN_DIR),
                    help=("override per_query/ input dir (e.g. "
                          "results/per_query_rltrain for leak-free calibration)"))
    ap.add_argument("--experts", default=",".join(RL_EXPERT_NAMES),
                    help=("comma-separated subset of RL experts to use for the "
                          "Δ>θ key mask (union mode). default = all 3."))
    ap.add_argument("--per_expert", action="store_true",
                    help="compute per-expert W: each task's prompts masked by "
                         "ONLY that task's own expert's Δ; output is per-layer "
                         "(N=3, d_out) tensor. merge_ablation.py auto-detects "
                         "2D shape and feeds W_i to expert i during kttrunc.")
    args = ap.parse_args()
    IN_DIR = Path(args.in_dir)

    sel_names = [s.strip() for s in args.experts.split(",") if s.strip()]
    bad = [n for n in sel_names if n not in RL_EXPERT_NAMES]
    if bad:
        raise ValueError(f"unknown expert names: {bad} (choose from {RL_EXPERT_NAMES})")

    if args.per_expert:
        # Each task's prompts → that task's own expert's mask → that expert's W
        # task ↔ expert pairing: ['coding','tool','memory'] ↔ ['coding_rl','tool_rl','memory_rl']
        if sel_names != list(RL_EXPERT_NAMES):
            raise ValueError("--per_expert requires the full default expert set "
                              "(coding_rl,tool_rl,memory_rl); got " + str(sel_names))
        print(f"[init] PER-EXPERT mode: each task masked by its own expert's Δ")
        seqs, prompt_lens, masks, owner_idx = build_per_expert_sequences_and_masks(
            TASKS, args.threshold, expert_names_per_task=list(RL_EXPERT_NAMES))
    else:
        print(f"[init] UNION mode: experts {sel_names} pooled (max of Δ over experts)")
        seqs, prompt_lens, masks = build_sequences_and_masks(
            TASKS, args.threshold, expert_names_subset=sel_names)
        owner_idx = None
    masks = [m.to(args.device) for m in masks]

    model = AutoModelForCausalLM.from_pretrained(
        _resolve_model_path(BASE), torch_dtype=torch.bfloat16,
        device_map={"": args.device}).eval()

    n_accums = len(RL_EXPERT_NAMES) if args.per_expert else 1
    state = {"mask": None, "ans_start": 0, "ans_end": 0, "owner": 0}
    linear_modules = []
    accums: list[dict[str, torch.Tensor]] = [dict() for _ in range(n_accums)]
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_modules.append((name, module.out_features))
            for ai in range(n_accums):
                accums[ai][f"{name}.weight"] = torch.zeros(
                    module.out_features, dtype=torch.float32, device=args.device)
    print(f"[init] {len(linear_modules)} Linear modules  ×  {n_accums} accumulator(s)")

    def make_hook(key: str, d_out: int):
        accs = [a[key] for a in accums]
        def hook(module, inputs, output):
            if output.ndim != 3 or output.shape[-1] != d_out: return
            m = state["mask"]
            if m is None: return
            a, b = state["ans_start"], state["ans_end"]
            y = output[0, a:b, :].float().abs()
            if y.shape[0] != m.shape[0]: return
            accs[state["owner"]].add_((m.unsqueeze(1) * y).sum(dim=0))
        return hook

    handles = []
    for name, d_out in linear_modules:
        module = dict(model.named_modules())[name]
        handles.append(module.register_forward_hook(make_hook(f"{name}.weight", d_out)))

    t0 = time.time()
    per_owner_key = [0] * n_accums; per_owner_pos = [0] * n_accums
    with torch.no_grad():
        for qi, (seq, pl, m) in enumerate(zip(seqs, prompt_lens, masks), start=1):
            L = int(seq.shape[0])
            if L <= pl: continue
            inp = seq.unsqueeze(0).to(args.device)
            owner = owner_idx[qi - 1] if owner_idx is not None else 0
            state["mask"] = m
            state["ans_start"] = pl - 1
            state["ans_end"] = L - 1
            state["owner"] = owner
            _ = model(inp)
            per_owner_key[owner] += int(m.sum().item())
            per_owner_pos[owner] += int(L - pl)
            state["mask"] = None
            if qi % 50 == 0 or qi == len(seqs):
                dt = time.time() - t0
                print(f"  {qi}/{len(seqs)}  ({dt:.0f}s)", flush=True)
    for h in handles: h.remove()
    if args.per_expert:
        for ai, en in enumerate(RL_EXPERT_NAMES):
            print(f"[done] {en}: {per_owner_key[ai]}/{per_owner_pos[ai]} key positions "
                  f"({per_owner_key[ai]/max(per_owner_pos[ai],1)*100:.2f}%)")
    else:
        print(f"[done] {per_owner_key[0]}/{per_owner_pos[0]} key positions "
              f"({per_owner_key[0]/max(per_owner_pos[0],1)*100:.2f}%)")

    # Per-layer: additive ε + normalize to median ≈ 1; per accumulator
    out_np: dict[str, np.ndarray] = {}
    layer_keys = list(accums[0].keys())
    for k in layer_keys:
        per_ai = []
        for ai in range(n_accums):
            w = accums[ai][k].cpu().numpy()
            med = float(np.median(w))
            if med <= 0:
                w_safe = np.ones_like(w)
            else:
                ε_l = med * args.eps_scale
                w_safe = (w + ε_l) / (med + ε_l)
            per_ai.append(w_safe.astype(np.float32))
        if n_accums == 1:
            out_np[k] = per_ai[0]               # 1D (d_out,)
        else:
            out_np[k] = np.stack(per_ai, axis=0) # 2D (N=3, d_out) — order = RL_EXPERT_NAMES

    print(f"[transform] threshold={args.threshold}  eps_scale={args.eps_scale}  "
          f"per_expert={args.per_expert}  (per-layer ε = eps_scale · median)")

    out_path = Path(args.out)
    np.savez_compressed(out_path, **out_np)
    print(f"[save] {out_path}  ({sum(a.nbytes for a in out_np.values())/1e6:.1f} MB)")

    print(f"\n=== sample layers dynamic range (post-ε safe) ===")
    sample_keys = sorted(out_np.keys())[:5] + sorted(out_np.keys())[-3:]
    for key in sample_keys:
        w = out_np[key]
        if w.ndim == 1:
            p50 = float(np.percentile(w, 50)); p99 = float(np.percentile(w, 99))
            print(f"  {key:<55}  d={w.shape[-1]:>6}  "
                  f"p50={p50:.3f}  p99/p50={p99/max(p50,1e-12):.2f}  "
                  f"max/p50={w.max()/max(p50,1e-12):.2f}")
        else:
            print(f"  {key:<55}  shape={tuple(w.shape)}")
            for ai, en in enumerate(RL_EXPERT_NAMES):
                wi = w[ai]
                p50 = float(np.percentile(wi, 50)); p99 = float(np.percentile(wi, 99))
                print(f"    [{en}]  p50={p50:.3f}  p99/p50={p99/max(p50,1e-12):.2f}  "
                      f"max/p50={wi.max()/max(p50,1e-12):.2f}")


if __name__ == "__main__":
    main()
