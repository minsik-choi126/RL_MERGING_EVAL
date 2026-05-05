"""3-way RL expert merge with optional KT-Polar (Key-Token-weighted polar
alignment) on lm_head.

Modes:
  no_polar          : base + Σ_i α_i · truncate_k(τ_i)   (same as 90energy_trunc_renorm)
  polar_all         : standard 90%-energy truncation + polar alignment + per-expert
                        renorm on ALL 2D layers (= merge.py per_tv_renorm).
                        Ablation baseline — isolates value of W over polar.
  polar_lm          : above + UNWEIGHTED polar alignment on lm_head
  ktpolar_lm        : row-wise BLEND of raw and polar on lm_head,
                        β(v) = (α_v − α_min)/(α_max − α_min),
                        τ_final[v] = β · τ_raw[v] + (1−β) · τ_polar[v]
  ktpolar_all       : row-wise BLEND on ALL 2D layers, using per-layer W
                        from results/ktpolar/W_activation.npz
  kttrunc_polar     : KT-Truncation on ALL 2D layers — weighted SVD
                        (truncate W_l^(1/2)·τ_i, unweight) chooses which
                        rank-k subspace is kept. Standard polar + per-expert
                        renorm follow. Addresses the alignment-analysis
                        finding that standard SVD drops high-W rows.

For every 2D layer:
  1. τ_i = W_expert_i - W_base
  2. SVD(τ_i) → top-k by 90% energy threshold
  3. k-truncated τ_i
  4. Per-expert renorm: α_i = ‖τ_i‖ / ‖τ_i^(k)‖, τ_i^(renorm) = α_i · τ_i^(k)

The extra polar step for lm_head (polar_lm / ktpolar_lm):
  a. Stack U_i (left bases), V_i (right bases), Σ_i per expert (after step 3)
  b. U_cat = concat(U_i), V_cat = concat(V_i)
  c. [weighted] U_hat = polar_factor(W_sqrt[:, None] * U_cat)  |  V_hat = polar_factor(V_cat)
  d. Aligned expert reconstruction in weighted/ original row space, renormed per expert

1D keys: TA mean (same as baseline).
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "deps"))
from kt_merge_helpers import (                                       # type: ignore
    compute_task_vector, get_2d_key_set, load_state_dict, save_model,
    _resolve_model_path,
)

# Default paths assume KT_merge/{scripts,outputs} layout. Override via CLI.
ROOT = _HERE.parent
OUT_DIR = ROOT / "outputs" / "merges"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALPHA_FILE = ROOT / "outputs" / "alphas.npz"
W_ACT_FILE = ROOT / "outputs" / "W_activation.npz"

# Defaults — overridable via CLI (`--base_model`, `--experts <name>=<path>...`)
# Default config: Qwen3-1.7B + 3 RL experts symlinked under ../models/
BASE = "Qwen/Qwen3-1.7B"
RL_EXPERTS = {
    "ifeval": str(ROOT / "models" / "ifeval"),
    "math":   str(ROOT / "models" / "math"),
    "lucy": str(ROOT / "models" / "lucy"),
}

LM_HEAD_KEY_CANDIDATES = ["lm_head.weight"]  # Qwen3 standard


def polar_factor(X: torch.Tensor) -> torch.Tensor:
    """Nearest orthogonal matrix via SVD: U @ Vh from SVD(X)."""
    U, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return U @ Vh


def kttrunc_per_expert(tau: torch.Tensor, W_row: torch.Tensor | None,
                         energy: float, device: str,
                         *, W_col: torch.Tensor | None = None):
    """Weighted-SVD truncation under a (D_row, D_col) diagonal weighting on
    opposite axes.

        Y       = D_row^(1/2) · τ · D_col^(1/2)
        Y       = U_y · Σ_y · V_y^T
        K       = smallest k s.t. Σ_{j≤k} σ_{y,j}² ≥ energy · Σ σ_{y,j}²
        Y^(K)   = U_y[:,:K] · diag(σ_y[:K]) · V_y[:K,:]
        τ^(K,W) = D_row^(-1/2) · Y^(K) · D_col^(-1/2)

    Either side may be None — that side is skipped. Both None ⇒ standard SVD.
    Shape-mismatched W is silently dropped on its own side.
    """
    t = tau.to(device)
    fro_full = float(t.norm().item())
    d_out, d_in = t.shape
    max_k = int(min(d_out, d_in))

    Wr_sqrt = None
    if W_row is not None:
        Wr = W_row.to(device).clamp_min(1e-12)
        if Wr.numel() == d_out:
            Wr_sqrt = Wr.sqrt().reshape(d_out, 1)
    Wc_sqrt = None
    if W_col is not None:
        Wc = W_col.to(device).clamp_min(1e-12)
        if Wc.numel() == d_in:
            Wc_sqrt = Wc.sqrt().reshape(1, d_in)

    weighted = (Wr_sqrt is not None) or (Wc_sqrt is not None)
    if fro_full <= 1e-12:
        return torch.zeros_like(t), {
            "k": 0, "max_k": max_k, "energy_preserved": 0.0,
            "fro_full": 0.0, "fro_trunc": 0.0,
            "kt_row": Wr_sqrt is not None,
            "kt_col": Wc_sqrt is not None,
            "kt_weighted": weighted, "note": "zero_layer",
        }

    Y = t
    if Wr_sqrt is not None: Y = Wr_sqrt * Y
    if Wc_sqrt is not None: Y = Y * Wc_sqrt

    Uy, Sy, Vhy = torch.linalg.svd(Y, full_matrices=False)
    sigma2 = (Sy * Sy).cpu().numpy()
    cum = np.cumsum(sigma2) / max(sigma2.sum(), 1e-12)
    K = int(np.searchsorted(cum, energy) + 1)
    K = min(K, len(Sy))
    Y_k = (Uy[:, :K] * Sy[:K].unsqueeze(0)) @ Vhy[:K, :]

    τ_k = Y_k
    if Wr_sqrt is not None: τ_k = τ_k / Wr_sqrt
    if Wc_sqrt is not None: τ_k = τ_k / Wc_sqrt

    note = "weighted_svd" if weighted else "standard_svd"
    fro_trunc = float(τ_k.norm().item())
    del Uy, Sy, Vhy, Y, Y_k
    return τ_k, {
        "k": K, "max_k": max_k,
        "energy_preserved_weighted": float(cum[K - 1]),
        "fro_full": fro_full, "fro_trunc": fro_trunc,
        "kt_row": Wr_sqrt is not None,
        "kt_col": Wc_sqrt is not None,
        "kt_weighted": weighted, "note": note,
    }


def merge_kttrunc_polar(taus: list[torch.Tensor], W_row: torch.Tensor | None,
                          energy: float, device: str, use_polar: bool = True):
    """KT-Truncation per expert, [optional polar alignment], per-expert renorm.

    Pipeline when use_polar=True:
      1. τ_i^(K,W) = kttrunc_per_expert(τ_i, W_l)         per expert
      2. Re-SVD τ_i^(K,W) to get its left/right bases
      3. polar_factor on concat(U_i^KW) and concat(V_i^KW)
      4. For each expert: τ_i_aligned = U_hat_block · diag(Σ) · V_hat_block^T
      5. Per-expert renorm α_i = ‖τ_i‖ / ‖τ_i_aligned‖
      6. Σ_i α_i · τ_i_aligned

    Pipeline when use_polar=False (kttrunc_only):
      1. τ_i^(K,W) per expert
      2. Per-expert renorm α_i = ‖τ_i‖ / ‖τ_i^(K,W)‖
      3. Σ_i α_i · τ_i^(K,W)
    """
    per_expert = []
    for tau in taus:
        τ_kw, stats = kttrunc_per_expert(tau, W_row, energy, device)
        if stats["k"] == 0:
            per_expert.append({"k": 0, "stats": stats, "tau_kw": None})
            continue
        if use_polar:
            # Re-SVD of τ_kw to get its own basis
            U, S, Vh = torch.linalg.svd(τ_kw, full_matrices=False)
            thr = S[0].abs() * 1e-6
            K_eff = int((S > thr).sum().item())
            K_eff = max(1, min(K_eff, stats["k"]))
            per_expert.append({
                "k": K_eff, "stats": stats,
                "U": U[:, :K_eff].contiguous(),
                "S": S[:K_eff].contiguous(),
                "V": Vh[:K_eff, :].T.contiguous(),
                "fro_full": stats["fro_full"], "tau_kw": τ_kw,
            })
            del U, S, Vh
        else:
            # no polar: keep tau_kw directly
            per_expert.append({
                "k": stats["k"], "stats": stats,
                "fro_full": stats["fro_full"], "tau_kw": τ_kw,
            })
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    valid = [e for e in per_expert if e["k"] > 0]
    d_out, d_in = taus[0].shape
    if not valid:
        return torch.zeros(d_out, d_in, dtype=torch.float32, device=device), per_expert

    merged = torch.zeros(d_out, d_in, dtype=torch.float32, device=device)

    if use_polar:
        U_cat = torch.cat([e["U"] for e in valid], dim=1)
        V_cat = torch.cat([e["V"] for e in valid], dim=1)
        U_hat = polar_factor(U_cat)
        V_hat = polar_factor(V_cat)
        del U_cat, V_cat
        col_idx = 0
        for e in per_expert:
            if e["k"] == 0: continue
            ki = e["k"]
            U_hat_block = U_hat[:, col_idx: col_idx + ki]
            V_hat_block = V_hat[:, col_idx: col_idx + ki]
            col_idx += ki
            tau_aligned = U_hat_block @ torch.diag(e["S"]) @ V_hat_block.T
            fro_aligned = float(tau_aligned.norm().item())
            alpha = e["fro_full"] / fro_aligned if fro_aligned > 1e-12 else 1.0
            merged += alpha * tau_aligned
            e["stats"]["alpha"] = alpha
            e["stats"]["fro_aligned"] = fro_aligned
            e["stats"]["aligned_polar"] = True
            del tau_aligned
        del U_hat, V_hat
    else:
        # No polar: renorm each tau_kw and sum
        for e in per_expert:
            if e["k"] == 0: continue
            tau_kw = e["tau_kw"]
            fro_trunc = float(tau_kw.norm().item())
            alpha = e["fro_full"] / fro_trunc if fro_trunc > 1e-12 else 1.0
            merged += alpha * tau_kw
            e["stats"]["alpha"] = alpha
            e["stats"]["aligned_polar"] = False

    # Free cached bases / intermediate tensors
    for e in per_expert:
        for fld in ("U", "S", "V", "tau_kw"):
            if fld in e: del e[fld]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return merged, per_expert


def svd_trunc_energy(tau: torch.Tensor, energy: float):
    """Adaptive-rank SVD at `energy` cutoff. Returns U,S,Vh, k_eff, stats."""
    t = tau
    fro_full = float(t.norm().item())
    max_k = int(min(t.shape))
    if fro_full <= 1e-12:
        return None, None, None, 0, {
            "k": 0, "max_k": max_k, "energy_preserved": 0.0,
            "alpha": 1.0, "fro_full": 0.0, "fro_trunc": 0.0,
            "note": "zero_layer",
        }
    U, S, Vh = torch.linalg.svd(t, full_matrices=False)
    sigma2 = (S * S).cpu().numpy()
    cum = np.cumsum(sigma2) / max(sigma2.sum(), 1e-12)
    k_eff = int(np.searchsorted(cum, energy) + 1)
    k_eff = min(k_eff, len(S))
    return U, S, Vh, k_eff, {
        "k": k_eff, "max_k": max_k,
        "energy_preserved": float(cum[k_eff - 1]),
        "fro_full": fro_full,
    }


def truncate_renorm(tau: torch.Tensor, energy: float, device: str):
    """Standard path (used for all non-polar 2D layers)."""
    t = tau.to(device)
    fro_full = float(t.norm().item())
    max_k = int(min(t.shape))
    if fro_full <= 1e-12:
        return torch.zeros_like(t), {"k": 0, "max_k": max_k,
            "energy_preserved": 0.0, "alpha": 1.0,
            "fro_full": 0.0, "fro_trunc": 0.0, "note": "zero_layer"}
    U, S, Vh = torch.linalg.svd(t, full_matrices=False)
    sigma2 = (S * S).cpu().numpy()
    cum = np.cumsum(sigma2) / max(sigma2.sum(), 1e-12)
    k_eff = int(np.searchsorted(cum, energy) + 1)
    k_eff = min(k_eff, len(S))
    Uk = U[:, :k_eff]; Sk = S[:k_eff]; Vk = Vh[:k_eff, :]
    recon = (Uk * Sk.unsqueeze(0)) @ Vk
    fro_trunc = float(recon.norm().item())
    alpha = fro_full / fro_trunc if fro_trunc > 1e-12 else 1.0
    return recon * alpha, {"k": k_eff, "max_k": max_k,
        "energy_preserved": float(cum[k_eff - 1]), "alpha": alpha,
        "fro_full": fro_full, "fro_trunc": fro_trunc}


def merge_lm_head_polar(taus: list[torch.Tensor], energy: float,
                          device: str, W_row: torch.Tensor | None):
    """Per-expert SVD shared between raw path and polar path (no duplicate SVD).

    W_row=None:
        Unweighted polar alignment only.
    W_row is a vector:
        Row-wise BLEND of Σᵢ τ_raw_i (no-polar) and Σᵢ τ_polar_i:
            β(v) = (W(v) − W_min)/(W_max − W_min) ∈ [0,1]
            τ_merged[v,:] = β(v)·Σᵢ τ_raw_i[v,:] + (1−β(v))·Σᵢ τ_polar_i[v,:]
    """
    # Single SVD per expert; cache Uk, Sk, Vk on device.
    per_expert = []
    for tau in taus:
        t = tau.to(device)
        U, S, Vh, k, stats = svd_trunc_energy(t, energy)
        if k == 0:
            per_expert.append({"k": 0, "stats": stats})
            del t
            continue
        Uk = U[:, :k].contiguous()
        Sk = S[:k].contiguous()
        Vk = Vh[:k, :].T.contiguous()  # (d_in, k)
        # Raw-path reconstruction inline (no second SVD)
        recon = (Uk * Sk.unsqueeze(0)) @ Vk.T
        fro_trunc = float(recon.norm().item())
        alpha_raw = stats["fro_full"] / fro_trunc if fro_trunc > 1e-12 else 1.0
        tau_raw = recon * alpha_raw
        stats["alpha"] = alpha_raw; stats["fro_trunc"] = fro_trunc
        per_expert.append({"k": k, "stats": stats, "U": Uk, "S": Sk, "V": Vk,
                             "fro_full": stats["fro_full"], "tau_raw": tau_raw})
        # Free intermediates
        del U, S, Vh, t, recon
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    valid = [e for e in per_expert if e["k"] > 0]
    if not valid:
        d_out, d_in = taus[0].shape
        return torch.zeros(d_out, d_in, dtype=torch.float32, device=device), per_expert

    # Polar-path alignment
    U_cat = torch.cat([e["U"] for e in valid], dim=1)
    V_cat = torch.cat([e["V"] for e in valid], dim=1)
    U_hat = polar_factor(U_cat)
    V_hat = polar_factor(V_cat)
    del U_cat, V_cat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    d_out, d_in = valid[0]["tau_raw"].shape
    tau_polar_sum = torch.zeros(d_out, d_in, dtype=torch.float32, device=device)
    tau_raw_sum   = torch.zeros(d_out, d_in, dtype=torch.float32, device=device)
    col_idx = 0
    for e in per_expert:
        if e["k"] == 0:
            continue
        ki = e["k"]
        U_hat_block = U_hat[:, col_idx: col_idx + ki]
        V_hat_block = V_hat[:, col_idx: col_idx + ki]
        col_idx += ki
        tau_aligned = U_hat_block @ torch.diag(e["S"]) @ V_hat_block.T
        fro_aligned = float(tau_aligned.norm().item())
        alpha_polar = e["fro_full"] / fro_aligned if fro_aligned > 1e-12 else 1.0
        tau_polar_sum += alpha_polar * tau_aligned
        tau_raw_sum   += e["tau_raw"]
        # Per-expert polar stats
        e["stats"]["alpha_polar"] = alpha_polar
        e["stats"]["fro_aligned"] = fro_aligned
        del tau_aligned
    del U_hat, V_hat
    # Release cached U/S/V/tau_raw to free VRAM for next layer
    for e in per_expert:
        for fld in ("U", "S", "V", "tau_raw"):
            if fld in e:
                del e[fld]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if W_row is None:
        for e in per_expert:
            if e["k"] > 0:
                e["stats"]["blend_mode"] = "polar_only"
        return tau_polar_sum, per_expert

    # Row-wise blend
    W = W_row.to(device).float()
    W_min = float(W.min().item()); W_max = float(W.max().item())
    if W_max - W_min < 1e-12:
        β = torch.zeros_like(W)
    else:
        β = (W - W_min) / (W_max - W_min)
    β_col = β.unsqueeze(1)
    tau_blended = β_col * tau_raw_sum + (1.0 - β_col) * tau_polar_sum
    del tau_raw_sum, tau_polar_sum
    for e in per_expert:
        if e["k"] > 0:
            e["stats"]["blend_mode"] = "row_blend"
            e["stats"]["beta_min"] = W_min
            e["stats"]["beta_max"] = W_max
    return tau_blended, per_expert


def find_lm_head_key(sd: dict) -> str | None:
    for cand in LM_HEAD_KEY_CANDIDATES:
        if cand in sd:
            return cand
    # Fallback: any key shaped (vocab, hidden)
    for k, v in sd.items():
        if v.ndim == 2 and v.shape[0] > 30_000:
            return k
    return None


def merge_experts(expert_names: list[str], energy: float, device: str,
                   out_dir: Path, mode: str, alpha_variant: str = "α4_bin_soft",
                   w_file_override: str | None = None):
    print(f"[init] mode={mode}  α_variant={alpha_variant}"
          f"{'  w_file='+w_file_override if w_file_override else ''}")
    print(f"[init] loading base state_dict")
    base_sd = load_state_dict(_resolve_model_path(BASE))
    two_d = get_2d_key_set(base_sd)

    lm_head_key = find_lm_head_key(base_sd)
    print(f"[init] lm_head key = {lm_head_key}")

    # Load W source (for KT-Polar)
    W_row_lm = None          # W for lm_head (mode=ktpolar_lm uses α directly)
    W_act_per_layer = None   # dict: key → vector (for ktpolar_all)
    if mode == "ktpolar_lm":
        if not ALPHA_FILE.exists():
            raise FileNotFoundError(f"{ALPHA_FILE} missing. Run ktpolar_alpha.py first.")
        alphas_npz = np.load(ALPHA_FILE)
        if alpha_variant not in alphas_npz.files:
            raise KeyError(f"alpha variant {alpha_variant} not in {alphas_npz.files}")
        α = alphas_npz[alpha_variant].astype(np.float32)
        W_row_lm = torch.from_numpy(α)
        print(f"[init] lm_head W: mean={α.mean():.3f}  p99={np.percentile(α,99):.3f}  "
              f"max={α.max():.3f}  shape={α.shape}")
    elif mode in ("ktpolar_all", "kttrunc_polar", "kttrunc_only"):
        w_file = Path(w_file_override) if w_file_override else W_ACT_FILE
        if not w_file.exists():
            raise FileNotFoundError(f"{w_file} missing.")
        W_act_per_layer = dict(np.load(w_file))
        print(f"[init] W_activation loaded from {w_file}: {len(W_act_per_layer)} layers")
        # sanity
        for sample_key in list(W_act_per_layer.keys())[:3]:
            w = W_act_per_layer[sample_key]
            p50 = np.percentile(w, 50); p99 = np.percentile(w, 99)
            print(f"  {sample_key:<55}  p50={p50:.2e}  p99/p50={p99/max(p50,1e-12):.2f}")

    merged_sd: dict[str, torch.Tensor] = {}
    per_layer_stats: list[dict] = []

    # Load all expert state_dicts
    expert_sds = {}
    for name in expert_names:
        print(f"[init] loading expert {name}")
        expert_sds[name] = load_state_dict(_resolve_model_path(RL_EXPERTS[name]))

    # 1D / non-2D keys: TA mean for floats, base for non-floats
    for key, v in base_sd.items():
        if key in two_d:
            continue
        if v.is_floating_point():
            W_base_1d = v.float()
            tvs = []
            for name in expert_names:
                sd = expert_sds[name]
                if key in sd and sd[key].is_floating_point():
                    tvs.append(sd[key].float() - W_base_1d)
            merged_sd[key] = ((W_base_1d + sum(tvs) / len(tvs)).to(v.dtype)
                              if tvs else v.clone())
        else:
            merged_sd[key] = v.clone()

    keys_2d = sorted(two_d)
    print(f"[merge] processing {len(keys_2d)} 2D layers, energy≥{energy}")
    t0 = time.time()
    for li, key in enumerate(keys_2d):
        is_lm = (key == lm_head_key)
        W_base = base_sd[key].float().to(device)
        taus = [expert_sds[name][key].float().to(device) - W_base for name in expert_names]

        # Decide which path this layer takes + what W to use
        use_polar = False
        use_kttrunc = False
        W_for_layer = None
        if mode == "polar_lm" and is_lm:
            use_polar = True
        elif mode == "polar_all":
            use_polar = True  # W_for_layer stays None → unweighted polar
        elif mode == "ktpolar_lm" and is_lm:
            use_polar = True
            W_for_layer = W_row_lm
        elif mode == "ktpolar_all":
            use_polar = True
            if key in W_act_per_layer:
                W_for_layer = torch.from_numpy(W_act_per_layer[key]).float()
                if W_for_layer.numel() != W_base.shape[0]:
                    W_for_layer = None
                    use_polar = False
        elif mode in ("kttrunc_polar", "kttrunc_only"):
            use_kttrunc = True
            if key in W_act_per_layer:
                W_for_layer = torch.from_numpy(W_act_per_layer[key]).float()
                if W_for_layer.numel() != W_base.shape[0]:
                    W_for_layer = None   # falls back to standard trunc inside

        if use_kttrunc:
            merged_tau, stats_list = merge_kttrunc_polar(
                taus, W_for_layer, energy, device,
                use_polar=(mode == "kttrunc_polar"))
            for i, e in enumerate(stats_list):
                s = e["stats"]
                s.update({"name": key, "expert": expert_names[i],
                          "polar": True, "kttrunc": True,
                          "weighted": W_for_layer is not None})
                per_layer_stats.append(s)
            merged_sd[key] = (W_base + merged_tau).cpu()
        elif use_polar:
            merged_tau, stats_list = merge_lm_head_polar(
                taus, energy, device, W_for_layer)
            for i, e in enumerate(stats_list):
                s = e["stats"]
                s.update({"name": key, "expert": expert_names[i], "polar": True,
                          "weighted": W_for_layer is not None})
                per_layer_stats.append(s)
            merged_sd[key] = (W_base + merged_tau).cpu()
        else:
            merged_tau_sum = torch.zeros_like(W_base)
            for i, tau in enumerate(taus):
                tau_r, s = truncate_renorm(tau, energy, device)
                s["name"] = key; s["expert"] = expert_names[i]
                s["polar"] = False
                per_layer_stats.append(s)
                merged_tau_sum += tau_r
                del tau_r
            merged_sd[key] = (W_base + merged_tau_sum).cpu()
            del merged_tau_sum

        del W_base, taus
        if (li + 1) % 50 == 0 or li == len(keys_2d) - 1:
            dt = time.time() - t0
            eta = dt / (li + 1) * (len(keys_2d) - li - 1)
            print(f"  layer {li+1}/{len(keys_2d)}  ({dt:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[save] model → {out_dir}")
    save_model(BASE, merged_sd, str(out_dir))
    stats_path = out_dir / "layer_stats.json"
    json.dump({"experts": expert_names, "energy": energy, "mode": mode,
                "alpha_variant": alpha_variant if mode == "ktpolar_lm" else None,
                "per_layer": per_layer_stats}, open(stats_path, "w"), indent=2)
    print(f"[save] layer stats → {stats_path}")

    for name in expert_names:
        ks = [s["k"] for s in per_layer_stats if s["expert"] == name and s["k"] > 0]
        energies = [s.get("energy_preserved", s.get("energy_preserved_weighted"))
                     for s in per_layer_stats
                     if s["expert"] == name and s["k"] > 0]
        energies = [e for e in energies if e is not None]
        alphas = [s.get("alpha", 1.0) for s in per_layer_stats
                   if s["expert"] == name and s["k"] > 0]
        print(f"  {name}: k min={min(ks)} median={int(np.median(ks))} "
              f"mean={np.mean(ks):.1f} max={max(ks)}   "
              f"energy avg={np.mean(energies):.3f}  α avg={np.mean(alphas):.3f}")

    return {"out_dir": str(out_dir), "n_layers": len(keys_2d),
             "n_experts": len(expert_names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=None,
                    help='comma-separated expert groups; supports 2-way or 3-way. '
                         'default: single 3-way merge of all RL experts.')
    ap.add_argument("--energy", type=float, default=0.90)
    ap.add_argument("--mode",
                     choices=["no_polar", "polar_lm", "polar_all",
                              "ktpolar_lm", "ktpolar_all",
                              "kttrunc_polar", "kttrunc_only"],
                     required=True)
    ap.add_argument("--alpha_variant", default="α4_bin_soft",
                     help="alpha variant key in alphas.npz (for ktpolar_lm mode)")
    ap.add_argument("--w_file", default=None,
                     help="override W_activation file path (e.g. .../W_activation_pure.npz)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--base_model", default=None,
                    help="override base model path / HF id (default: hard-coded Qwen2.5-7B)")
    ap.add_argument("--experts", nargs="+", default=None,
                    help=("override RL experts as <name>=<path> pairs, e.g. "
                          "ifeval=models/ifeval math=models/math lucy=models/lucy"))
    ap.add_argument("--out_dir", default=None,
                    help="override output dir for merge (default: outputs/merges)")
    args = ap.parse_args()

    # Apply CLI overrides for base / experts / out_dir
    global BASE, RL_EXPERTS, OUT_DIR
    if args.base_model:
        BASE = args.base_model
    if args.experts:
        RL_EXPERTS = {}
        for spec in args.experts:
            if "=" not in spec:
                raise ValueError(f"--experts spec must be name=path, got: {spec}")
            name, path = spec.split("=", 1)
            RL_EXPERTS[name.strip()] = path.strip()
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[init] BASE = {BASE}")
    print(f"[init] EXPERTS = {RL_EXPERTS}")
    print(f"[init] OUT_DIR = {OUT_DIR}")

    if not (0.0 < args.energy <= 1.0):
        raise ValueError(f"--energy must be in (0, 1]; got {args.energy}")

    if args.tag is None:
        pct_str = f"{args.energy * 100:g}".replace(".", "p")
        suffix = {"no_polar": "no_polar", "polar_lm": "polar_lm",
                   "polar_all": "polar_all",
                   "ktpolar_lm": f"ktpolar_lm_{args.alpha_variant}",
                   "ktpolar_all": "ktpolar_all_act",
                   "kttrunc_polar": "kttrunc_polar_act",
                   "kttrunc_only": "kttrunc_only_act"}[args.mode]
        args.tag = f"k{pct_str}e_{suffix}"
    print(f"[init] energy={args.energy}  tag={args.tag}  mode={args.mode}")

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[init] SVD device = {device}")

    if args.pairs:
        pairs = [tuple(spec.split(",")) for spec in args.pairs]
    else:
        pairs = [tuple(RL_EXPERTS.keys())]  # default: single 3-way merge
    print(f"[init] {len(pairs)} merge group(s): {pairs}")

    manifest = []
    for pair in pairs:
        pair_key = "__".join(pair)
        print(f"\n========== merge {pair_key} (mode={args.mode}) ==========")
        out_dir = OUT_DIR / f"{pair_key}_{args.tag}"
        if args.dry_run:
            manifest.append({"pair": pair_key, "out_dir": str(out_dir), "status": "dry_run"})
            continue
        if (out_dir / "model.safetensors").exists() or list(out_dir.glob("*.safetensors")):
            print(f"[skip] {out_dir} has safetensors already")
            manifest.append({"pair": pair_key, "out_dir": str(out_dir), "status": "skipped"})
            continue
        info = merge_experts(list(pair), args.energy, device, out_dir,
                              args.mode, args.alpha_variant, args.w_file)
        info["pair"] = pair_key; info["mode"] = args.mode
        manifest.append(info)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mf_path = OUT_DIR / f"manifest_{args.tag}.json"
    json.dump(manifest, open(mf_path, "w"), indent=2)
    print(f"\n[done] manifest → {mf_path}")


if __name__ == "__main__":
    main()
