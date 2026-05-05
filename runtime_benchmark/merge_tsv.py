"""TSV (Task Singular Vectors, CVPR 2025) merge — single-GPU.

For 2D layers: per-expert SVD top-k (k = max(1, d_min/N) by default), block-
concatenate U/V/Σ across N experts, polar-orthogonalize the (d_out × N·k) and
(N·k × d_in) blocks, reconstruct merged τ via 5-product (u_u·v_u)·diag(Σ)·(u_v·v_v).
1D layers: rolling mean of task vectors. Final: base + α·merged_τ.

Self-contained — extracted from merge_baseline.run_tsv but does not import
from the rest of the project.
"""
from __future__ import annotations
import argparse, gc, sys, time
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from helpers import (                              # type: ignore
    setup_cache, load_state_dict, save_model, get_numeric_keys,
)


def run_tsv(base_sd, expert_sds, alpha: float, k: Optional[int],
             sv_reduction: Optional[float], device: str) -> dict:
    N = len(expert_sds)
    keys = get_numeric_keys(base_sd, expert_sds)
    if k is None:
        sv_reduction = sv_reduction or (1.0 / N)

    merged_tv = {}
    with torch.no_grad():
        for key in tqdm(keys, desc="TSV Merge"):
            tv_list = [sd[key].float().to(device) - base_sd[key].float().to(device)
                       for sd in expert_sds]
            shape = tv_list[0].shape

            if len(shape) >= 2:
                M, D = shape[0], shape[1]
                min_dim = min(M, D)

                U0, S0, Vh0 = torch.linalg.svd(tv_list[0], full_matrices=False)
                k_per = (min(k, min_dim) if k
                          else max(1, int(min_dim * sv_reduction)))
                if N * k_per > min_dim:
                    k_per = max(1, min_dim // N)

                sum_u = torch.zeros(M, min_dim, device=device)
                sum_s = torch.zeros(min_dim, device=device)
                sum_v = torch.zeros(min_dim, D, device=device)
                sum_u[:, :k_per] = U0[:, :k_per]
                sum_s[:k_per] = S0[:k_per]
                sum_v[:k_per, :] = Vh0[:k_per, :]
                del U0, S0, Vh0
                for i, tv in enumerate(tv_list[1:], 1):
                    U, S, Vh = torch.linalg.svd(tv, full_matrices=False)
                    sum_u[:, i * k_per:(i + 1) * k_per] = U[:, :k_per]
                    sum_s[i * k_per:(i + 1) * k_per] = S[:k_per]
                    sum_v[i * k_per:(i + 1) * k_per, :] = Vh[:k_per, :]
                    del U, S, Vh

                u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                merged_tv[key] = torch.linalg.multi_dot((
                    u_u, v_u, torch.diag(sum_s), u_v, v_v,
                )).cpu()
                del sum_u, sum_s, sum_v, u_u, v_u, u_v, v_v
            else:
                # 1D rolling mean
                result = tv_list[0].clone()
                for i, tv in enumerate(tv_list[1:], 1):
                    result.add_((tv - result) / (i + 1))
                merged_tv[key] = result.cpu()

    final_sd = {}
    for key in base_sd:
        if key in merged_tv:
            final_sd[key] = base_sd[key].float() + alpha * merged_tv[key]
        else:
            final_sd[key] = base_sd[key]
    return final_sd


def main():
    ap = argparse.ArgumentParser(description="TSV merge (1 GPU)")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--expert_coding", required=True)
    ap.add_argument("--expert_tool", required=True)
    ap.add_argument("--expert_memory", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=None,
                    help="Fixed singular-values per expert (default: 1/N of min_dim)")
    ap.add_argument("--sv_reduction", type=float, default=None)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()

    setup_cache(args.cache_dir)

    print("=" * 64)
    print(f"  TSV merge   start={time.strftime('%F %T')}")
    print(f"  device={args.device}  alpha={args.alpha}")
    print(f"  base   : {args.base_model}")
    print(f"  coding : {args.expert_coding}")
    print(f"  tool   : {args.expert_tool}")
    print(f"  memory : {args.expert_memory}")
    print(f"  out    : {args.out_dir}")
    print("=" * 64)

    print("\n[1/3] Loading state dicts...")
    t_load = time.time()
    base_sd = load_state_dict(args.base_model)
    expert_sds = [
        load_state_dict(args.expert_coding),
        load_state_dict(args.expert_tool),
        load_state_dict(args.expert_memory),
    ]
    print(f"  load took {time.time()-t_load:.1f}s")

    print("\n[2/3] Merging (TSV)...")
    t_merge = time.time()
    final_sd = run_tsv(base_sd, expert_sds,
                          alpha=args.alpha, k=args.k,
                          sv_reduction=args.sv_reduction, device=args.device)
    print(f"  merge wall: {time.time()-t_merge:.1f}s")

    del expert_sds
    gc.collect()

    print("\n[3/3] Saving...")
    save_model(args.base_model, final_sd, args.out_dir)
    print(f"\n[done] {args.out_dir}")


if __name__ == "__main__":
    main()
