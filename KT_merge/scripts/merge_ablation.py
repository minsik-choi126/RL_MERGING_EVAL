"""Component-wise ablation of KT-Truncation merging.

Isolates contributions of three orthogonal axes:
  - truncate : none | svd (Frobenius 90% energy) | kt (W-weighted 90% energy)
  - polar    : align U/V bases via polar_factor over [U_1|U_2|U_3] cat
  - renorm   : per-expert α_i = ‖τ_i‖_F / ‖τ_aligned_i‖_F restoration

7 variants (all 2D layers; 1D layers always TA mean):

    | tag                  | truncate | polar | renorm |
    |----------------------|----------|-------|--------|
    | naive                | none     | no    | no     |
    | svd                  | svd      | no    | no     |
    | svd_polar            | svd      | yes   | no     |
    | svd_polar_renorm     | svd      | yes   | yes    |  (= existing polar_all)
    | kt                   | kt       | no    | no     |
    | kt_polar             | kt       | yes   | no     |
    | kt_polar_renorm      | kt       | yes   | yes    |  (= existing kttrunc_polar)

State_dicts are loaded once at startup and reused across every requested
variant; merged_sd is rebuilt per variant. Default W file:
W_activation_positionkey_0.1.npz (the proposed-method W).

1D layers are ALWAYS merged with TA mean across every variant — the
ablation isolates the 2D axes (truncate / polar / renorm).

Usage:
    # build all 7 variants in one run (sequential, ~3 h on 1 GPU)
    python merge_ablation.py --variants all

    # build a single variant
    python merge_ablation.py --variants kt_polar_renorm

    # subset
    python merge_ablation.py --variants naive,svd,svd_polar_renorm,kt_polar_renorm

Output (default --out_root ../outputs/ablation/):
    outputs/ablation/{variant}/
        ├── model.safetensors
        ├── layer_stats.json
        ├── config.json, tokenizer*, ...
    outputs/ablation/manifest.json
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "deps"))
from kt_merge_helpers import (                                       # type: ignore
    get_2d_key_set, load_state_dict, save_model, _resolve_model_path,
)

# Reuse helpers from sibling merge_ktpolar.py
sys.path.insert(0, str(_HERE))
from merge_ktpolar import polar_factor, kttrunc_per_expert           # type: ignore

# KT_merge layout: scripts/, deps/, data/, outputs/
ROOT = _HERE.parent
OUT_DIR = ROOT / "outputs" / "ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_W_FILE = ROOT / "outputs" / "W_activation_positionkey_0.1.npz"

# Defaults — overridable via CLI (--base_model / --experts)
BASE = "Qwen/Qwen3-1.7B"
RL_EXPERTS = {
    "ifeval": str(ROOT / "models" / "ifeval"),
    "math":   str(ROOT / "models" / "math"),
    "coding": str(ROOT / "models" / "coding"),
}

# W_activation is collected from nn.Linear outputs. Embedding matrices are 2D
# tensors in the state_dict, but they are not Linear modules and do not have a
# matching W row-importance vector. Keep them on the fixed non-Linear TA-mean
# path instead of letting KT variants fall back to unweighted SVD.
NON_LINEAR_2D_KEYS = {"model.embed_tokens.weight"}
MERGE_IMPL_VERSION = "2026-04-27-nonlinear2d-ta-mean"

VARIANTS: dict[str, dict] = {
    "naive":            {"truncate": "none", "polar": False, "renorm": False},
    "svd":              {"truncate": "svd",  "polar": False, "renorm": False},
    "svd_polar":        {"truncate": "svd",  "polar": True,  "renorm": False},
    "svd_polar_renorm": {"truncate": "svd",  "polar": True,  "renorm": True},
    "kt":               {"truncate": "kt",   "polar": False, "renorm": False, "kt_mode": "row"},
    "kt_polar":         {"truncate": "kt",   "polar": True,  "renorm": False, "kt_mode": "row"},
    "kt_polar_renorm":  {"truncate": "kt",   "polar": True,  "renorm": True,  "kt_mode": "row"},
    "ktcol":               {"truncate": "kt", "polar": False, "renorm": False, "kt_mode": "col"},
    "ktcol_polar":         {"truncate": "kt", "polar": True,  "renorm": False, "kt_mode": "col"},
    "ktcol_polar_renorm":  {"truncate": "kt", "polar": True,  "renorm": True,  "kt_mode": "col"},
    "kt2s":               {"truncate": "kt", "polar": False, "renorm": False, "kt_mode": "2s"},
    "kt2s_polar":         {"truncate": "kt", "polar": True,  "renorm": False, "kt_mode": "2s"},
    "kt2s_polar_renorm":  {"truncate": "kt", "polar": True,  "renorm": True,  "kt_mode": "2s"},
}

# 1D-layer treatment is FIXED to TA mean across all 7 variants so the
# ablation isolates the 2D axes (truncate / polar / renorm). Naive's 1D
# is therefore "mean" too — by design — even though its 2D is unscaled sum.


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer ablation merge core
# ─────────────────────────────────────────────────────────────────────────────

def _svd_energy_truncate(tau: torch.Tensor, energy: float):
    """Return (Uk, Sk, Vk_T_transposed_to_(d_in,k), tau_k, K, energy_pres).

    Frobenius-energy adaptive rank: smallest k s.t. Σσ²[:k] ≥ energy · Σσ²
    """
    U, S, Vh = torch.linalg.svd(tau, full_matrices=False)
    sigma2 = (S * S).cpu().numpy()
    cum = np.cumsum(sigma2) / max(sigma2.sum(), 1e-12)
    K = int(np.searchsorted(cum, energy) + 1)
    K = min(K, len(S))
    Uk = U[:, :K].contiguous()
    Sk = S[:K].contiguous()
    Vk = Vh[:K, :].T.contiguous()                     # (d_in, K)
    tau_k = (Uk * Sk.unsqueeze(0)) @ Vk.T
    return Uk, Sk, Vk, tau_k, K, float(cum[K - 1])


def _resvd_for_polar(tau_k: torch.Tensor, K_hint: int):
    """Re-SVD a reconstructed (rank-K) tau back into bases in ORIGINAL space.

    Used for the kt path: τ^(K,W) is in original space but the U/V from
    weighted SVD are in weighted space — polar alignment across experts
    needs original-space orthonormal bases, so we re-SVD τ^(K,W) here.
    Cost is small because rank ≤ K_hint.
    """
    U, S, Vh = torch.linalg.svd(tau_k, full_matrices=False)
    if S.numel() == 0:
        return U, S, Vh.T, 0
    thr = S[0].abs() * 1e-6
    K_eff = int((S > thr).sum().item())
    K_eff = max(1, min(K_eff, K_hint))
    return U[:, :K_eff].contiguous(), S[:K_eff].contiguous(), \
           Vh[:K_eff, :].T.contiguous(), K_eff


def _split_per_expert(W: torch.Tensor | None, N: int):
    if W is None: return [None] * N
    if W.ndim == 2:
        if W.shape[0] != N:
            raise ValueError(f"per-expert W has {W.shape[0]} rows but {N} experts")
        return [W[i] for i in range(N)]
    return [W] * N


def merge_one_layer(
    taus: list[torch.Tensor], energy: float, device: str, *,
    truncate: str, polar: bool, renorm: bool, kt_mode: str = "row",
    W_row: torch.Tensor | None = None,
    W_col: torch.Tensor | None = None,
):
    """Single 2D-layer merge under a (truncate, polar, renorm, kt_mode) configuration."""
    N = len(taus)
    fro_full = [float(t.norm().item()) for t in taus]
    d_out, d_in = taus[0].shape

    # Mask out unused W per kt_mode
    if truncate != "kt":
        W_row = W_col = None
    elif kt_mode == "row": W_col = None
    elif kt_mode == "col": W_row = None
    elif kt_mode == "2s":  pass
    else: raise ValueError(f"unknown kt_mode: {kt_mode}")

    Wrow_per = _split_per_expert(W_row, N)
    Wcol_per = _split_per_expert(W_col, N)
    W_per_expert = Wrow_per   # backward-compat alias for kt-row path

    # ── Naive sum: no truncation, no polar, no renorm ──────────────────────
    if truncate == "none":
        merged = torch.zeros_like(taus[0])
        for t in taus:
            merged += t
        stats = [{"truncate": "none", "polar": False, "renorm": False,
                   "fro_full": fro_full[i], "k": min(taus[i].shape),
                   "max_k": min(taus[i].shape), "alpha": 1.0}
                  for i in range(N)]
        return merged, stats

    # ── Per-expert truncation (cache U,S,V in ORIGINAL row space) ──────────
    per_expert: list[dict] = []
    for i, tau in enumerate(taus):
        if truncate == "svd":
            Uk, Sk, Vk, tau_k, K, energy_pres = _svd_energy_truncate(tau, energy)
            stats_i = {"truncate": "svd", "k": K, "max_k": int(min(tau.shape)),
                        "energy_preserved": energy_pres,
                        "fro_full": fro_full[i], "fro_trunc": float(tau_k.norm().item())}
        elif truncate == "kt":
            tau_kw, ks = kttrunc_per_expert(
                tau, Wrow_per[i], energy, device, W_col=Wcol_per[i],
            )
            ks["kt_mode"] = kt_mode
            if ks["k"] == 0:
                per_expert.append({"k": 0, "stats": ks, "fro_full": fro_full[i]})
                continue
            tau_k = tau_kw                               # already in original space
            if polar:
                # Re-SVD only if we need orthonormal bases for cross-expert polar.
                # Skipping this saves a (d_out × d_in) SVD on lm_head for variant 4.
                Uk, Sk, Vk, K_eff = _resvd_for_polar(tau_kw, ks["k"])
                stats_i = {"truncate": "kt", **ks,
                            "k_eff_post_resvd": K_eff,
                            "fro_full": fro_full[i],
                            "fro_trunc": float(tau_k.norm().item())}
            else:
                Uk = Sk = Vk = None
                stats_i = {"truncate": "kt", **ks,
                            "fro_full": fro_full[i],
                            "fro_trunc": float(tau_k.norm().item())}
        else:
            raise ValueError(f"unknown truncate: {truncate}")

        per_expert.append({
            "k": (Sk.shape[0] if Sk is not None else ks["k"]),
            "stats": stats_i,
            "U": Uk, "S": Sk, "V": Vk, "tau_k": tau_k,
            "fro_full": fro_full[i],
        })

    valid = [e for e in per_expert if e.get("k", 0) > 0]
    if not valid:
        return torch.zeros(d_out, d_in, dtype=torch.float32, device=device), \
               [e.get("stats", {}) for e in per_expert]

    merged = torch.zeros(d_out, d_in, dtype=torch.float32, device=device)

    # ── Path 1: no polar — sum truncated taus, optionally renormed ─────────
    if not polar:
        for e in per_expert:
            if e.get("k", 0) == 0:
                continue
            tau_k = e["tau_k"]
            if renorm:
                fro_t = float(tau_k.norm().item())
                alpha = e["fro_full"] / fro_t if fro_t > 1e-12 else 1.0
                merged += alpha * tau_k
                e["stats"]["alpha"] = alpha
            else:
                merged += tau_k
                e["stats"]["alpha"] = 1.0
            e["stats"]["polar"] = False
            e["stats"]["renorm"] = renorm
    # ── Path 2: polar alignment, optionally renormed ───────────────────────
    else:
        U_cat = torch.cat([e["U"] for e in valid], dim=1)
        V_cat = torch.cat([e["V"] for e in valid], dim=1)
        U_hat = polar_factor(U_cat)
        V_hat = polar_factor(V_cat)
        del U_cat, V_cat
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
            if renorm:
                alpha = e["fro_full"] / fro_a if fro_a > 1e-12 else 1.0
            else:
                alpha = 1.0
            merged += alpha * tau_aligned
            e["stats"]["alpha"] = alpha
            e["stats"]["fro_aligned"] = fro_a
            e["stats"]["polar"] = True
            e["stats"]["renorm"] = renorm
            del tau_aligned
        del U_hat, V_hat

    # cleanup cached tensors
    for e in per_expert:
        for fld in ("U", "S", "V", "tau_k"):
            if fld in e:
                del e[fld]

    return merged, [e.get("stats", {}) for e in per_expert]


# ─────────────────────────────────────────────────────────────────────────────
# Per-variant driver
# ─────────────────────────────────────────────────────────────────────────────

def run_variant(
    variant: str, expert_names: list[str], expert_sds: dict[str, dict],
    base_sd: dict, two_d: set, keys_2d: list[str],
    W_act_per_layer: dict | None,
    energy: float, device: str, out_dir: Path,
    W_col_per_layer: dict | None = None,
):
    flags = VARIANTS[variant]
    truncate, polar, renorm = flags["truncate"], flags["polar"], flags["renorm"]
    kt_mode = flags.get("kt_mode", "row")
    needs_Wrow = (truncate == "kt") and (kt_mode in ("row", "2s"))
    needs_Wcol = (truncate == "kt") and (kt_mode in ("col", "2s"))
    if needs_Wrow and W_act_per_layer is None:
        raise RuntimeError(f"variant {variant} (kt_mode={kt_mode}) requires --w_file")
    if needs_Wcol and W_col_per_layer is None:
        raise RuntimeError(f"variant {variant} (kt_mode={kt_mode}) requires --w_col_file")
    print(f"\n========= variant={variant}  "
          f"truncate={truncate}  polar={polar}  renorm={renorm}  "
          f"kt_mode={kt_mode}  1D=mean (fixed) =========")

    merged_sd: dict[str, torch.Tensor] = {}

    # 1D / non-2D — TA mean for all variants (held constant for ablation purity).
    # Non-float (e.g., embed indices, masks): copy from base unchanged.
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

    per_layer_stats: list[dict] = []
    n_fallback_layers = 0   # count of 2D layers where kt path silently fell back to SVD
    t0 = time.time()
    for li, key in enumerate(keys_2d):
        W_base = base_sd[key].float().to(device)
        taus = [expert_sds[n][key].float().to(device) - W_base for n in expert_names]

        d_out_l, d_in_l = W_base.shape

        def _resolve_W(side_per_layer, axis_size):
            if side_per_layer is None: return None, False
            arr = side_per_layer.get(key)
            if arr is None: return None, True
            if arr.ndim == 1 and arr.shape[0] == axis_size:
                return torch.from_numpy(arr).float(), False
            if arr.ndim == 2 and arr.shape[1] == axis_size:
                return torch.from_numpy(arr).float(), False
            return None, True

        W_row, fb_r = _resolve_W(W_act_per_layer if needs_Wrow else None, d_out_l)
        W_col, fb_c = _resolve_W(W_col_per_layer if needs_Wcol else None, d_in_l)
        if fb_r or fb_c:
            n_fallback_layers += 1

        merged_tau, stats_list = merge_one_layer(
            taus, energy, device,
            truncate=truncate, polar=polar, renorm=renorm, kt_mode=kt_mode,
            W_row=W_row, W_col=W_col,
        )
        for i, s in enumerate(stats_list):
            s = dict(s) if s else {}
            s.update({"name": key, "expert": expert_names[i], "variant": variant})
            per_layer_stats.append(s)
        merged_sd[key] = (W_base + merged_tau).cpu()

        del W_base, taus, merged_tau
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (li + 1) % 50 == 0 or li == len(keys_2d) - 1:
            dt = time.time() - t0
            eta = dt / (li + 1) * (len(keys_2d) - li - 1)
            print(f"  [{variant}] layer {li+1}/{len(keys_2d)}  "
                  f"({dt:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[save] {variant} → {out_dir}")
    save_model(BASE, merged_sd, str(out_dir))
    json.dump(
        {"variant": variant, "flags": flags, "experts": expert_names,
         "energy": energy, "n_fallback_layers": n_fallback_layers,
         "n_2d_layers": len(keys_2d), "oned_policy": "mean",
         "merge_impl_version": MERGE_IMPL_VERSION,
         "per_layer": per_layer_stats},
        open(out_dir / "layer_stats.json", "w"), indent=2,
    )

    # Quick per-expert summary
    for name in expert_names:
        ks = [s.get("k", 0) for s in per_layer_stats
              if s.get("expert") == name and s.get("k", 0) > 0]
        alphas = [s.get("alpha", 1.0) for s in per_layer_stats
                  if s.get("expert") == name and s.get("k", 0) > 0]
        if ks:
            print(f"  {name}: k median={int(np.median(ks))}  α avg={np.mean(alphas):.3f}")
        else:
            print(f"  {name}: (no rank-k > 0 layers — naive variant?)")
    if needs_W:
        print(f"  [kt] W-fallback layers: {n_fallback_layers}/{len(keys_2d)} "
              f"(layers without matching W → unweighted SVD)")

    del merged_sd, per_layer_stats
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"variant": variant, "out_dir": str(out_dir),
             "n_layers": len(keys_2d), "n_experts": len(expert_names),
             "n_fallback_layers": n_fallback_layers,
             "oned_policy": "mean",
             "flags": flags}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global BASE, RL_EXPERTS   # declared up front so argparse defaults read them
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="all",
                     help="comma-separated variant names, or 'all'. "
                          f"choices: {','.join(VARIANTS)}")
    ap.add_argument("--energy", type=float, default=0.90)
    ap.add_argument("--w_file", default=str(DEFAULT_W_FILE),
                     help="W_activation row file (d_out per layer) for kt row/2s variants")
    ap.add_argument("--w_col_file", default=None,
                     help="W_activation column file (d_in per layer) for kt col/2s variants")
    ap.add_argument("--experts", default=",".join(RL_EXPERTS),
                     help="comma-separated subset of experts (default: all 3)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_root", default=str(OUT_DIR),
                     help="root directory for variant subdirs")
    ap.add_argument("--skip_existing", action="store_true",
                     help="skip variants whose out_dir already has *.safetensors")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--base_model", default=None,
                     help="override BASE model path/HF id (default: hard-coded)")
    ap.add_argument("--expert_paths", nargs="+", default=None,
                     help=("override expert paths as <name>=<path>, e.g. "
                           "ifeval=models/ifeval math=models/math coding=models/coding"))
    args = ap.parse_args()

    # Apply CLI overrides for BASE / RL_EXPERTS
    if args.base_model:
        BASE = args.base_model
    if args.expert_paths:
        RL_EXPERTS = {}
        for spec in args.expert_paths:
            if "=" not in spec:
                raise ValueError(f"--expert_paths spec must be name=path, got: {spec}")
            name, path = spec.split("=", 1)
            RL_EXPERTS[name.strip()] = path.strip()
    print(f"[init] BASE = {BASE}")
    print(f"[init] EXPERTS = {RL_EXPERTS}")

    # Resolve variants
    if args.variants == "all":
        variants = list(VARIANTS)
    else:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise ValueError(f"unknown variants: {bad}.  choices: {list(VARIANTS)}")

    expert_names = [e.strip() for e in args.experts.split(",") if e.strip()]
    bad = [e for e in expert_names if e not in RL_EXPERTS]
    if bad:
        raise ValueError(f"unknown experts: {bad}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[init] variants = {variants}")
    print(f"[init] experts  = {expert_names}")
    print(f"[init] energy   = {args.energy}")
    print(f"[init] out_root = {out_root}")
    print(f"[init] 1D policy= mean (fixed for all variants)")

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[init] device   = {device}")

    # ── Plan output dirs & decide which variants need to actually run ───────
    plan = []
    for variant in variants:
        out_dir = out_root / variant
        already = (out_dir / "model.safetensors").exists() or list(out_dir.glob("*.safetensors"))
        if args.skip_existing and already:
            stats_path = out_dir / "layer_stats.json"
            existing_version = None
            if stats_path.exists():
                try:
                    existing_version = json.load(open(stats_path)).get("merge_impl_version")
                except Exception:
                    existing_version = None
            if existing_version == MERGE_IMPL_VERSION:
                print(f"[skip] {variant}: {out_dir} already populated")
                continue
            print(f"[rebuild] {variant}: existing output predates merge_impl_version={MERGE_IMPL_VERSION}")
        plan.append((variant, out_dir))

    if args.dry_run:
        print("\n[dry-run] would build:")
        for variant, out_dir in plan:
            print(f"  {variant:<20} → {out_dir}")
        return

    if not plan:
        print("[done] nothing to do (all variants already exist)")
        return

    # ── Load state_dicts ONCE (shared across all requested variants) ────────
    print(f"\n[load] base state_dict")
    t0 = time.time()
    base_sd = load_state_dict(_resolve_model_path(BASE))
    print(f"  done in {time.time()-t0:.1f}s")

    expert_sds = {}
    for name in expert_names:
        t0 = time.time()
        print(f"[load] expert {name}")
        expert_sds[name] = load_state_dict(_resolve_model_path(RL_EXPERTS[name]))
        print(f"  done in {time.time()-t0:.1f}s")

    two_d = get_2d_key_set(base_sd) - NON_LINEAR_2D_KEYS
    keys_2d = sorted(two_d)
    print(f"[init] {len(keys_2d)} 2D layers")

    # ── Load W (row/col) only if any planned variant needs them ─────────────
    kt_modes = {VARIANTS[v].get("kt_mode", "row")
                for v, _ in plan if VARIANTS[v]["truncate"] == "kt"}
    needs_Wrow = any(m in ("row", "2s") for m in kt_modes)
    needs_Wcol = any(m in ("col", "2s") for m in kt_modes)

    def _load_W(label, path_str):
        if path_str is None:
            raise FileNotFoundError(f"{label} required but not set")
        p = Path(path_str)
        if not p.exists(): raise FileNotFoundError(f"{label} missing: {p}")
        print(f"[load] {label} from {p}")
        d = dict(np.load(p))
        print(f"  {len(d)} layers")
        for sample_key in list(d.keys())[:3]:
            w = d[sample_key]
            p50 = np.percentile(w, 50); p99 = np.percentile(w, 99)
            print(f"    {sample_key:<55}  shape={w.shape}  "
                  f"p50={p50:.2e}  p99/p50={p99/max(p50,1e-12):.2f}")
        return d

    W_act_per_layer = _load_W("w_file (row)", args.w_file) if needs_Wrow else None
    W_col_per_layer = _load_W("w_col_file (col)", args.w_col_file) if needs_Wcol else None

    # ── Run each variant sequentially ───────────────────────────────────────
    manifest = []
    t_all = time.time()
    for variant, out_dir in plan:
        info = run_variant(
            variant=variant, expert_names=expert_names,
            expert_sds=expert_sds, base_sd=base_sd, two_d=two_d, keys_2d=keys_2d,
            W_act_per_layer=W_act_per_layer,
            W_col_per_layer=W_col_per_layer,
            energy=args.energy, device=device, out_dir=out_dir,
        )
        manifest.append(info)
    dt_all = time.time() - t_all
    print(f"\n[done] built {len(manifest)} variants in {dt_all/60:.1f} min")

    mf_path = out_root / "manifest.json"
    json.dump({"experts": expert_names, "energy": args.energy,
               "oned_policy": "mean",
               "merge_impl_version": MERGE_IMPL_VERSION,
               "w_file":     args.w_file     if needs_Wrow else None,
               "w_col_file": args.w_col_file if needs_Wcol else None,
               "variants": manifest},
              open(mf_path, "w"), indent=2)
    print(f"[done] manifest → {mf_path}")


if __name__ == "__main__":
    main()
