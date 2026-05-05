"""KT-col-neg + polar + renorm merge for 3 RL experts on a single GPU.

Per 2D layer ℓ and expert i, with τ_i = W_expert_i - W_base and a per-expert
column weighting D_c (from --w_col_file):

    Y_i      = τ_i · D_c^{1/2}                     # weighted (col-only)
    Y_i      = U_y · Σ_y · V_y^T                   # full SVD
    K_i      = smallest k s.t. Σ σ²[:k] ≥ energy · Σ σ²
    τ_i^(K)  = (U_y[:,:K] · diag(σ_y[:K]) · V_y[:K]^T) · D_c^{-1/2}
    # Polar across experts in ORIGINAL row space:
    M_left_i = U_y[:,:K]                            # already orthonormal; no D_r
    M_right_i= V_y[:,:K]^T · D_c^{-1/2}             # (K × d_in)
    QR(M_right_i^T) → Q_r_i (d_in × K), R_r_i (K × K)
    middle_i = diag(σ_y[:K]) · R_r_i^T              # (K × K)
    SVD(middle_i)  → U_m_i, S_m_i, V_m_i           # tiny K×K SVD
    U_k_i    = M_left_i · U_m_i                     # (d_out × K) orthonormal
    V_k_i    = Q_r_i · V_m_i^T                      # (d_in × K)  orthonormal
    S_k_i    = S_m_i

    # Polar across experts:
    U_hat = polar(cat(U_k_i)),  V_hat = polar(cat(V_k_i))
    τ_aligned_i = U_hat[:, blocks_i] · diag(S_k_i) · V_hat[:, blocks_i]^T
    α_i         = ‖τ_i‖ / ‖τ_aligned_i‖              # per-expert renorm
    merged_τ    = Σ_i α_i · τ_aligned_i

The optimization vs the original 'merge_ablation.py' replaces the second
full-rank SVD (`_resvd_for_polar`) with QR+small-SVD, dropping the
lm_head dominant cost from SVD(d_out × d_in) to QR(d_in × K) + SVD(K × K).

Output saved as a HF-style merged checkpoint at --out_dir.
"""
from __future__ import annotations
import argparse, gc, sys, time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from helpers import (                          # type: ignore
    setup_cache, resolve_path, load_state_dict, save_model, get_numeric_keys,
)


# ─────────────────────────────────────────────────────────────────────────────
def polar_factor(X: torch.Tensor) -> torch.Tensor:
    """Nearest-orthogonal factor of X via thin SVD: U @ Vh."""
    U, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return U @ Vh


def kttrunc_col(tau: torch.Tensor, W_col: torch.Tensor | None,
                 energy: float, device: str):
    """Energy-truncated weighted SVD with column weighting only.
       Returns τ^(K) in original space + cache for downstream polar."""
    t = tau.to(device)
    fro_full = float(t.norm().item())
    d_out, d_in = t.shape
    max_k = int(min(d_out, d_in))

    Wc_sqrt = None
    if W_col is not None:
        Wc = W_col.to(device).clamp_min(1e-12)
        if Wc.numel() == d_in:
            Wc_sqrt = Wc.sqrt().reshape(1, d_in)

    if fro_full <= 1e-12:
        zero = torch.zeros_like(t)
        return zero, {"k": 0, "max_k": max_k, "fro_full": 0.0}, None

    Y = t * Wc_sqrt if Wc_sqrt is not None else t
    Uy, Sy, Vhy = torch.linalg.svd(Y, full_matrices=False)

    # K via cumulative energy on GPU (one .item() at end)
    sigma2 = Sy * Sy
    cum = sigma2.cumsum(0) / sigma2.sum().clamp_min(1e-12)
    K = int((cum < energy).sum().item()) + 1
    K = min(K, len(Sy))

    Uy_K = Uy[:, :K].contiguous()
    Sy_K = Sy[:K].contiguous()
    Vhy_K = Vhy[:K, :].contiguous()

    Y_k = (Uy_K * Sy_K.unsqueeze(0)) @ Vhy_K
    tau_k = Y_k / Wc_sqrt if Wc_sqrt is not None else Y_k

    info = {"k": K, "max_k": max_k, "fro_full": fro_full,
              "fro_trunc": float(tau_k.norm().item())}
    cache = {"Uy_K": Uy_K, "Sy_K": Sy_K, "Vhy_K": Vhy_K, "Wc_sqrt": Wc_sqrt}
    del Uy, Sy, Vhy, Y, Y_k
    return tau_k, info, cache


def resvd_via_qr(cache: dict):
    """Original-space orthonormal U,S,V from kttrunc_col cache via QR+small SVD.
       Cheap because Uy_K is already orthonormal (col-only weighting)."""
    Uy_K = cache["Uy_K"]      # (d_out, K) orthonormal in original-row space
    Sy_K = cache["Sy_K"]      # (K,)
    Vhy_K = cache["Vhy_K"]    # (K, d_in)  weighted-col space
    Wc_sqrt = cache["Wc_sqrt"]

    if Wc_sqrt is None:
        # No weighting — Uy_K and Vhy_K are already final.
        return Uy_K, Sy_K, Vhy_K.T.contiguous(), Sy_K.shape[0]

    M_right = Vhy_K / Wc_sqrt  # (K, d_in) — original-col space
    # M_right.T has shape (d_in, K)
    Q_r, R_r = torch.linalg.qr(M_right.T.contiguous(), mode="reduced")
    # middle = diag(Sy_K) · R_r.T   (K × K)
    middle = (R_r * Sy_K.unsqueeze(1)).T
    U_m, S_m, Vh_m = torch.linalg.svd(middle, full_matrices=False)
    if S_m.numel() == 0:
        return Uy_K[:, :0], S_m, Q_r[:, :0], 0

    thr = S_m[0].abs() * 1e-6
    K_eff = int((S_m > thr).sum().item())
    K_eff = max(1, min(K_eff, S_m.numel()))

    Uk = (Uy_K @ U_m[:, :K_eff]).contiguous()       # (d_out, K_eff)
    Sk = S_m[:K_eff].contiguous()
    Vk = (Q_r @ Vh_m[:K_eff, :].T).contiguous()      # (d_in, K_eff)
    return Uk, Sk, Vk, K_eff


def merge_one_layer(taus, W_col_per_expert, energy, device):
    """ktcol_polar_renorm path for a single 2D layer."""
    N = len(taus)
    fro_full = [float(t.norm().item()) for t in taus]
    d_out, d_in = taus[0].shape

    per_expert = []
    for i, tau in enumerate(taus):
        tau_kw, info, cache = kttrunc_col(tau, W_col_per_expert[i], energy, device)
        if info["k"] == 0 or cache is None:
            per_expert.append({"k": 0, "fro_full": fro_full[i]})
            continue
        Uk, Sk, Vk, K_eff = resvd_via_qr(cache)
        per_expert.append({"k": K_eff, "U": Uk, "S": Sk, "V": Vk,
                            "tau_k": tau_kw, "fro_full": fro_full[i]})
        del cache

    valid = [e for e in per_expert if e.get("k", 0) > 0]
    if not valid:
        return torch.zeros(d_out, d_in, dtype=torch.float32, device=device)

    U_cat = torch.cat([e["U"] for e in valid], dim=1)
    V_cat = torch.cat([e["V"] for e in valid], dim=1)
    U_hat = polar_factor(U_cat); V_hat = polar_factor(V_cat)
    del U_cat, V_cat

    merged = torch.zeros(d_out, d_in, dtype=torch.float32, device=device)
    col = 0
    for e in per_expert:
        if e.get("k", 0) == 0:
            continue
        ki = e["k"]
        U_blk = U_hat[:, col: col + ki]
        V_blk = V_hat[:, col: col + ki]
        col += ki
        tau_aligned = U_blk @ torch.diag(e["S"]) @ V_blk.T
        fro_a = float(tau_aligned.norm().item())
        alpha = e["fro_full"] / fro_a if fro_a > 1e-12 else 1.0
        merged += alpha * tau_aligned
        del tau_aligned

    for e in per_expert:
        for fld in ("U", "S", "V", "tau_k"):
            if fld in e:
                del e[fld]
    return merged


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="ktcol_polar_renorm merge (1 GPU)")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--expert_coding", required=True)
    ap.add_argument("--expert_tool", required=True)
    ap.add_argument("--expert_memory", required=True)
    ap.add_argument("--w_col_file", required=True,
                    help="W_col npz from extract_w.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--energy", type=float, default=0.90)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()

    setup_cache(args.cache_dir)

    print("=" * 64)
    print(f"  Ours (ktcol_polar_renorm) merge   start={time.strftime('%F %T')}")
    print(f"  device={args.device}  energy={args.energy}")
    print(f"  base   : {args.base_model}")
    print(f"  coding : {args.expert_coding}")
    print(f"  tool   : {args.expert_tool}")
    print(f"  memory : {args.expert_memory}")
    print(f"  w_col  : {args.w_col_file}")
    print(f"  out    : {args.out_dir}")
    print("=" * 64)

    print("\n[1/3] Loading state dicts...")
    t_load = time.time()
    base_sd = load_state_dict(args.base_model)
    expert_sds = {
        "coding_rl": load_state_dict(args.expert_coding),
        "tool_rl":   load_state_dict(args.expert_tool),
        "memory_rl": load_state_dict(args.expert_memory),
    }
    print(f"  load took {time.time()-t_load:.1f}s")

    print("\n[2/3] Loading W_col npz + 2D layer plan...")
    W_col_per_layer = dict(np.load(args.w_col_file, allow_pickle=True))
    keys_all = get_numeric_keys(base_sd, list(expert_sds.values()))
    keys_2d = [k for k in keys_all if base_sd[k].dim() == 2]
    keys_1d = [k for k in keys_all if base_sd[k].dim() != 2]
    print(f"  2D layers: {len(keys_2d)}    1D/non-2D: {len(keys_1d)}")
    print(f"  W_col covers {len(W_col_per_layer)} layers")

    print("\n[3/3] Merging...")
    merged_sd = {}
    expert_names = ["coding_rl", "tool_rl", "memory_rl"]

    # 1D / non-2D: TA mean (uniform expert avg)
    for k in keys_1d:
        if not base_sd[k].is_floating_point():
            merged_sd[k] = base_sd[k].clone()
            continue
        base1 = base_sd[k].float()
        tvs = [expert_sds[n][k].float() - base1 for n in expert_names if k in expert_sds[n]]
        merged_sd[k] = ((base1 + sum(tvs) / len(tvs)).to(base_sd[k].dtype)
                         if tvs else base_sd[k].clone())

    # Also copy any base keys we didn't touch yet
    for k, v in base_sd.items():
        if k not in keys_2d and k not in merged_sd:
            merged_sd[k] = v.clone()

    # 2D layers: ktcol_polar_renorm
    t_merge = time.time()
    for li, key in enumerate(keys_2d):
        W_base = base_sd[key].float().to(args.device)
        taus = [expert_sds[n][key].float().to(args.device) - W_base
                for n in expert_names]
        d_out, d_in = W_base.shape

        # Resolve per-expert W_col for this layer
        arr = W_col_per_layer.get(key)
        W_col_per_expert = [None, None, None]
        if arr is not None:
            if arr.ndim == 2 and arr.shape == (3, d_in):
                W_col_per_expert = [torch.from_numpy(arr[i]).float() for i in range(3)]
            elif arr.ndim == 1 and arr.shape[0] == d_in:
                W_col_per_expert = [torch.from_numpy(arr).float()] * 3

        merged_tau = merge_one_layer(taus, W_col_per_expert, args.energy, args.device)
        merged_sd[key] = (W_base + merged_tau).to(base_sd[key].dtype).cpu()
        del W_base, taus, merged_tau
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (li + 1) % 50 == 0 or li == len(keys_2d) - 1:
            dt = time.time() - t_merge
            eta = dt / (li + 1) * (len(keys_2d) - li - 1)
            print(f"  layer {li+1}/{len(keys_2d)}  ({dt:.0f}s elapsed, ETA {eta:.0f}s)")
    print(f"  merge wall: {time.time()-t_merge:.1f}s")

    del expert_sds
    gc.collect()

    print("\nSaving merged model...")
    save_model(args.base_model, merged_sd, args.out_dir)
    print(f"\n[done] {args.out_dir}")


if __name__ == "__main__":
    main()
