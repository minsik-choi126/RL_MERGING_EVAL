#!/usr/bin/env python3
"""
Unified Model Merging Script

모델 경로와 머징 기법만 선택해서 실행할 수 있는 통합 스크립트.

지원 머징 기법 (--method):
  task_arithmetic   : base + Σ λ_i·(expert_i - base)
  ties              : Trim-Elect Sign-Disjoint Merge
  dare              : DARE (Drop And REscale) + task_arithmetic or ties
  star              : STAR (Singular value Truncation And Rescaling, NAACL 2025)
  cart              : CART (Centered And Rank-Truncated, EMNLP 2024)
  tsv               : Task Singular Vectors (SVD 기반)
  fisher            : Fisher precision-weighted merge
  whitened          : Polar whitening + per-layer energy scaling
  whitened_noscale  : Polar whitening (스케일링 없음, ablation)
  per_tv_renorm     : Polar whitening + per-expert energy 복원
  capped_pertv      : Polar whitening + capped per-expert + global renorm
  energy_direction  : 위 변형들을 한 번에 생성 (실험용)
  global_whitened   : Global α rescaling (energy_direction의 (f) 변형 단독 실행)
  cat               : CAT (Conflict-Aware Task merging, activation 기반 conflict 제거)
  lot               : LOT (Least-squares Optimal Task merging, activation 기반 최적 머징)
  iso_c             : Iso-C (Isotropic Merging in Common Subspace, ICML 2025)
  iso_cts           : Iso-CTS (Isotropic Merging in Common & Task-Specific Subspaces, ICML 2025)
  ram               : RAM (Reinforced Agentic Merge) — overlap-aware averaging
  ram_plus          : RAM+ (RAM with overlap-aware rescaling, ARM-R-V2)
  svd_truncation    : SVD Truncation — task vector를 rank-k로 truncate 후 단순 합산
  rmt_whitened      : RMT auto-k + polar whitening + per-layer energy scaling
  rmt_per_tv        : RMT auto-k + polar whitening + per-expert energy 복원
  rmt_optimal       : RMT auto-k + polar whitening + Donoho-Gavish optimal shrinkage

기본 사용법:
  python merge.py --method task_arithmetic \\
      --base_model  Qwen/Qwen2.5-7B-Instruct \\
      --expert_models /path/to/model_a /path/to/model_b \\
      --save_dir    /path/to/output

  python merge.py --method whitened \\
      --base_model  Qwen/Qwen2.5-7B-Instruct \\
      --expert_models /path/to/model_a /path/to/model_b \\
      --save_dir    /path/to/output \\
      --k_list 128 256 512
"""

import argparse
import gc
import json
import math
import os
from collections import OrderedDict, defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

# 머지 결과물 저장 루트 검증 — 기본은 비활성 (빈 문자열).
# 특정 디렉토리로 제한하려면 환경변수 KT_SAVE_ROOT 설정.
import os as _os
REQUIRED_SAVE_ROOT = _os.environ.get("KT_SAVE_ROOT", "")

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ══════════════════════════════════════════════════════════════════════════════
# RMT (Random Matrix Theory) 유틸리티 — Gavish & Donoho (2014) optimal threshold
# ══════════════════════════════════════════════════════════════════════════════

def _marchenko_pastur_median(beta: float) -> float:
    """
    Marchenko-Pastur 분포의 median을 수치적으로 계산.
    beta = m/n (aspect ratio, m <= n).
    """
    from scipy.integrate import quad
    from scipy.optimize import brentq

    lam_minus = (1 - math.sqrt(beta)) ** 2
    lam_plus = (1 + math.sqrt(beta)) ** 2

    def mp_density(x):
        if x <= lam_minus or x >= lam_plus:
            return 0.0
        return math.sqrt((lam_plus - x) * (x - lam_minus)) / (2 * math.pi * beta * x)

    def cdf_at(t):
        val, _ = quad(mp_density, lam_minus, t)
        return val

    # CDF(median) = 0.5 를 만족하는 median 찾기
    med = brentq(lambda t: cdf_at(t) - 0.5, lam_minus + 1e-10, lam_plus - 1e-10)
    return med


def _omega(beta: float) -> float:
    """
    Gavish & Donoho (2014)의 ω(β) 함수.
    Optimal hard threshold = σ_med * ω(β)
    where σ_med = median singular value (noise level estimator).

    ω(β) = λ*(β) / μ_β 에서:
    - μ_β = sqrt(marchenko_pastur_median(β))
    - λ*(β) = sqrt(2(β+1) + 8β / ((β+1) + sqrt(β²+14β+1)))
    """
    mu_beta = math.sqrt(_marchenko_pastur_median(beta))

    # optimal singular value threshold (Theorem in Gavish & Donoho 2014)
    lam_star = math.sqrt(
        2 * (beta + 1) + 8 * beta / ((beta + 1) + math.sqrt(beta**2 + 14 * beta + 1))
    )
    return lam_star / mu_beta


# 캐시: beta → omega 값 (동일 shape 레이어가 많으므로)
_omega_cache: Dict[float, float] = {}


def gavish_donoho_threshold(S: torch.Tensor, m: int, n: int) -> int:
    """
    Gavish-Donoho optimal hard threshold를 적용하여 signal rank k를 결정.

    Args:
        S: singular values (1D tensor, 내림차순)
        m, n: 원래 행렬의 shape (m <= n으로 가정 안 함, 내부에서 처리)

    Returns:
        k: signal로 판별된 singular value 개수 (threshold를 넘는 것들)
    """
    # beta = min(m,n) / max(m,n) ≤ 1
    m_eff, n_eff = min(m, n), max(m, n)
    beta = m_eff / n_eff

    beta_key = round(beta, 6)
    if beta_key not in _omega_cache:
        _omega_cache[beta_key] = _omega(beta)
    omega_val = _omega_cache[beta_key]

    # noise level 추정: median singular value
    sigma_med = S[len(S) // 2].item()

    # optimal threshold
    threshold = sigma_med * omega_val

    # threshold를 넘는 singular value 수
    k = int((S > threshold).sum().item())
    return max(k, 1)  # 최소 1개는 유지


# ══════════════════════════════════════════════════════════════════════════════
# 공통 유틸리티
# ══════════════════════════════════════════════════════════════════════════════

def setup_cache(cache_dir: Optional[str] = None):
    """HuggingFace 캐시 경로 설정."""
    if cache_dir:
        hf_home = cache_dir
    else:
        # 기존 환경 변수 우선, 없으면 기본값
        candidates = [
            os.environ.get("HF_HOME"),
            "/mnt/ddn/merit/dev/251223_weight_merging/cache/huggingface",
        ]
        hf_home = next((c for c in candidates if c and os.path.isdir(os.path.dirname(c))), None)
        if hf_home is None:
            return  # 캐시 설정 없이 진행

    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))


def _resolve_path(model_path: str) -> str:
    """로컬 디렉토리면 그대로, HF 캐시에서 찾거나 HF Hub에서 다운로드."""
    if os.path.isdir(model_path):
        return model_path
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = os.path.join(hf_home, "hub")
    slug = "models--" + model_path.replace("/", "--")
    snapshots = os.path.join(hub_dir, slug, "snapshots")
    if os.path.isdir(snapshots):
        snaps = sorted(Path(snapshots).iterdir(), key=lambda p: p.stat().st_mtime)
        if snaps:
            return str(snaps[-1])
    return model_path


def load_state_dict(model_path: str) -> dict:
    """safetensors 우선, 없으면 HF AutoModel로 로드 (float32, CPU)."""
    resolved = _resolve_path(model_path)
    sf_files = sorted(Path(resolved).glob("*.safetensors"))
    if sf_files:
        from safetensors.torch import load_file
        print(f"  Loading safetensors ({len(sf_files)} shards): {resolved}")
        sd = {}
        for sf in sf_files:
            sd.update(load_file(str(sf), device="cpu"))
        return sd

    print(f"  Loading via HF AutoModel: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    sd = model.state_dict()
    del model
    gc.collect()
    return sd


def save_model(base_model_path: str, merged_sd: dict, out_dir: str):
    """HuggingFace 포맷으로 저장 (safetensors + tokenizer)."""
    print(f"  Saving → {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float32, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.load_state_dict(merged_sd, strict=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    del model
    gc.collect()
    tok = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tok.save_pretrained(out_dir)


def get_numeric_keys(base_sd: dict, expert_sds: List[dict]) -> List[str]:
    """공통 numeric 키만 추출 (int64, uint8 제외)."""
    common = set(base_sd.keys())
    for sd in expert_sds:
        common &= set(sd.keys())
    return sorted(
        k for k in common
        if base_sd[k].dtype not in (torch.int64, torch.uint8)
    )


# ══════════════════════════════════════════════════════════════════════════════
# 머징 기법 구현
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# 1. Task Arithmetic
# ──────────────────────────────────────────────────────────────────────────────

def run_task_arithmetic(base_sd: dict, expert_sds: List[dict], lambdas: List[float]) -> dict:
    """
    base + Σ λ_i · (expert_i - base)
    """
    keys = get_numeric_keys(base_sd, expert_sds)
    final_sd = {}

    with torch.no_grad():
        for key in tqdm(keys, desc="Task Arithmetic"):
            merged = base_sd[key].float().clone()
            for i, sd in enumerate(expert_sds):
                merged.add_(lambdas[i] * (sd[key].float() - base_sd[key].float()))
            final_sd[key] = merged

    # 비-numeric 키는 base에서 복사
    for key in base_sd:
        if key not in final_sd:
            final_sd[key] = base_sd[key]

    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 2. TIES Merging
# ──────────────────────────────────────────────────────────────────────────────

def _state_dict_to_vector(sd: dict, remove_keys: set) -> torch.Tensor:
    items = sorted((k, v) for k, v in sd.items() if k not in remove_keys)
    # float32로 통일: bfloat16/float16 safetensors 로드 시 mixed-dtype cat 방지
    return torch.nn.utils.parameters_to_vector(
        [v.float().reshape(-1) for _, v in items]
    )


def _vector_to_state_dict(vector: torch.Tensor, ref_sd: dict, remove_keys: set) -> dict:
    ref = OrderedDict(sorted((k, v.clone()) for k, v in ref_sd.items() if k not in remove_keys))
    torch.nn.utils.vector_to_parameters(vector, ref.values())
    return ref


def _topk_mask(M: torch.Tensor, K: float) -> torch.Tensor:
    """각 행에서 상위 K% 값만 유지."""
    if K > 1:
        K /= 100
    if M.dim() == 1:
        M = M.unsqueeze(0)
    n, d = M.shape
    k = max(1, d - int(d * K))
    kth, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    return M * (M.abs() >= kth)


def _resolve_sign(T: torch.Tensor, method: str) -> torch.Tensor:
    if method == "mass":
        signs = torch.sign(T.sum(dim=0))
    elif method == "normfrac":
        norms = torch.norm(T, dim=1, keepdim=True)
        nf = T ** 2 / (norms ** 2 + 1e-12)
        signs = torch.sign(T[nf.argmax(dim=0), torch.arange(T.shape[1])])
    elif method == "normmass":
        norms = torch.norm(T, dim=1, keepdim=True)
        nf = T ** 2 / (norms ** 2 + 1e-12)
        signs = (T.sign() * nf.abs()).sum(dim=0).sign()
    else:
        raise ValueError(f"Unknown sign_method: {method}")
    # 0인 위치는 전체 다수결 부호로 채움
    majority = torch.sign(signs.sum())
    signs[signs == 0] = majority
    return signs


def _disjoint_merge(T: torch.Tensor, signs: torch.Tensor, func: str) -> torch.Tensor:
    agg = func.split("-")[-1]
    mask = torch.where(signs.unsqueeze(0) > 0, T > 0, T < 0)
    selected = T * mask
    if agg == "mean":
        count = (selected != 0).sum(dim=0).float()
        return selected.sum(dim=0) / count.clamp(min=1)
    elif agg == "sum":
        return selected.sum(dim=0)
    elif agg == "max":
        return selected.abs().max(dim=0)[0] * signs
    else:
        raise ValueError(f"Unknown merge_func: {func}")


def run_ties(
    base_sd: dict, expert_sds: List[dict],
    lamda: float, density: float, sign_method: str, merge_func: str,
    device: str,
) -> dict:
    """
    Trim → Elect Sign → Disjoint Merge → base + λ·merged_tv
    """
    int_keys = {k for k in base_sd if base_sd[k].dtype in (torch.int64, torch.uint8)}

    flat_base = _state_dict_to_vector(base_sd, int_keys).to(device)
    flat_experts = torch.vstack([
        _state_dict_to_vector(sd, int_keys).to(device) for sd in expert_sds
    ])

    tvs = flat_experts - flat_base
    del flat_experts

    print(f"  TRIM: top-{density*100:.0f}%")
    trimmed = _topk_mask(tvs, density)
    del tvs

    print(f"  ELECT SIGN: {sign_method}")
    signs = _resolve_sign(trimmed, sign_method)

    print(f"  DISJOINT MERGE: {merge_func}")
    if "dis" in merge_func:
        merged_tv = _disjoint_merge(trimmed, signs, merge_func)
    elif merge_func == "sum":
        merged_tv = trimmed.sum(dim=0)
    else:  # "mean"
        merged_tv = trimmed.mean(dim=0)

    del trimmed

    final_flat = flat_base + lamda * merged_tv
    del flat_base, merged_tv

    final_sd = _vector_to_state_dict(final_flat.cpu(), base_sd, int_keys)
    del final_flat

    for k in int_keys:
        final_sd[k] = base_sd[k]

    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 3. DARE (Drop And REscale) Merging
# ──────────────────────────────────────────────────────────────────────────────

def _dare_mask_tensor(
    delta: torch.Tensor,
    mask_rate: float,
    use_rescale: bool,
    mask_strategy: str,
) -> torch.Tensor:
    """
    DARE 핵심 연산: delta 텐서에서 mask_rate 비율의 원소를 드롭하고
    use_rescale=True이면 1 / (1 - mask_rate) 로 재스케일.

    mask_strategy:
      "random"    : Bernoulli 샘플링으로 랜덤 드롭
      "magnitude" : 절댓값 기준 하위 mask_rate 비율 드롭
    """
    assert 0.0 <= mask_rate <= 1.0, f"mask_rate must be in [0, 1], got {mask_rate}"

    if mask_strategy == "random":
        mask = torch.bernoulli(torch.full_like(delta, mask_rate))
        masked = delta * (1 - mask)
    elif mask_strategy == "magnitude":
        original_shape = delta.shape
        flat = delta.flatten()
        num_mask = int(len(flat) * mask_rate)
        if num_mask == 0:
            return delta.clone()
        kth_val, _ = flat.abs().kthvalue(k=num_mask, dim=0, keepdim=True)
        keep_mask = flat.abs() >= kth_val
        masked = (flat * keep_mask).reshape(original_shape)
    else:
        raise ValueError(f"Unknown mask_strategy: {mask_strategy}")

    if use_rescale and mask_rate < 1.0:
        masked = masked / (1 - mask_rate)

    return masked


def run_dare(
    base_sd: dict,
    expert_sds: List[dict],
    lambdas: List[float],
    weight_mask_rate: float,
    use_rescale: bool,
    mask_strategy: str,
    merge_method: str,
    # TIES 전용 파라미터 (merge_method == "ties" 일 때만 사용)
    ties_lamda: float = 1.0,
    ties_density: float = 0.2,
    ties_sign_method: str = "mass",
    ties_merge_func: str = "dis-mean",
    device: str = "cpu",
) -> dict:
    """
    DARE (Drop And REscale) Merging.
    출처: https://github.com/yule-BUAA/MergeLM

    Step 1: 각 expert의 delta = expert - base 에 _dare_mask_tensor 적용
            (mask_rate 비율 드롭 + 선택적 재스케일)
    Step 2: 희소화된 expert로 task_arithmetic 또는 ties 머징 수행

    Args:
        base_sd          : 베이스 모델 state dict
        expert_sds       : 전문가 모델 state dict 리스트
        lambdas          : 전문가별 스케일 계수 (task_arithmetic 사용 시)
        weight_mask_rate : 드롭할 delta 원소 비율 (0.0~1.0, 논문 기본값 0.9)
        use_rescale      : True면 생존 원소를 1/(1-mask_rate)로 재스케일
        mask_strategy    : "random" | "magnitude"
        merge_method     : "task_arithmetic" | "ties"
        ties_*           : ties 파라미터 (merge_method=="ties" 일 때만 사용)
        device           : ties 계산 디바이스
    """
    keys = get_numeric_keys(base_sd, expert_sds)

    # ── Step 1: DARE 희소화 ──
    sparsified_sds = []
    for i, expert_sd in enumerate(expert_sds):
        print(f"  [DARE] Expert {i}: mask_rate={weight_mask_rate}, "
              f"strategy={mask_strategy}, rescale={use_rescale}")
        sparse_sd = {}
        with torch.no_grad():
            for key in tqdm(keys, desc=f"    Expert {i} DARE"):
                delta = expert_sd[key].float() - base_sd[key].float()
                sparse_delta = _dare_mask_tensor(delta, weight_mask_rate, use_rescale, mask_strategy)
                sparse_sd[key] = base_sd[key].float() + sparse_delta
        # 비-numeric 키는 그대로 복사
        for key in expert_sd:
            if key not in sparse_sd:
                sparse_sd[key] = expert_sd[key]
        sparsified_sds.append(sparse_sd)

    # ── Step 2: 희소화된 expert로 머징 ──
    print(f"  [DARE] Post-merge method: {merge_method}")
    if merge_method == "task_arithmetic":
        return run_task_arithmetic(base_sd, sparsified_sds, lambdas)
    elif merge_method == "ties":
        return run_ties(
            base_sd, sparsified_sds,
            ties_lamda, ties_density, ties_sign_method, ties_merge_func, device,
        )
    else:
        raise ValueError(f"Unknown merge_method for DARE: {merge_method}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. STAR (Singular value Truncation And Rescaling)
# ──────────────────────────────────────────────────────────────────────────────

def _star_compress_tensor(delta: torch.Tensor, eta: float) -> torch.Tensor:
    """
    STAR 핵심 연산: 2D delta 행렬을 SVD 후 nuclear norm의 최소 eta%를 커버하는
    rank까지 truncate하고, 잔여 singular values의 합이 원래 합과 동일하도록 rescale 후 재구성.

    수식:
      sum_tot = Σ sᵢ
      rank_remain = min r s.t. Σᵢ₌₁ʳ sᵢ ≥ eta/100 · sum_tot
      scaled_s = (sum_tot / Σᵢ₌₁ʳ sᵢ) · s[:r]
      output = U[:, :r] @ diag(scaled_s) @ Vt[:r, :]

    1D 텐서(bias, layernorm 등)는 그대로 반환.
    """
    if delta.dim() < 2:
        return delta.clone()

    u, s, vt = torch.linalg.svd(delta, full_matrices=False)

    sum_tot_s = torch.sum(s)
    if sum_tot_s == 0:
        return delta.clone()

    # nuclear norm의 eta% 커버하는 최소 rank 탐색
    cumulative_s = torch.cumsum(s, dim=0)
    rank_remain = torch.searchsorted(cumulative_s, sum_tot_s * eta / 100.0).item() + 1
    rank_remain = max(1, min(rank_remain, s.shape[0]))

    # Truncate
    u_t  = u[:, :rank_remain]
    s_t  = s[:rank_remain]
    vt_t = vt[:rank_remain, :]

    # Rescale: Σ scaled_sᵢ = sum_tot_s
    sum_remain = torch.sum(s_t)
    scaled_s = (sum_tot_s / sum_remain) * s_t

    return u_t @ torch.diag(scaled_s) @ vt_t


def run_star(
    base_sd: dict,
    expert_sds: List[dict],
    eta: float,
    lambdas: List[float],
) -> dict:
    """
    STAR (Singular value Truncation And Rescaling) Merging.
    출처: https://github.com/IBM/STAR (NAACL 2025)

    Step 1: 각 expert의 delta에 _star_compress_tensor 적용
            (nuclear norm eta% 커버 rank까지 truncate + singular value rescale)
    Step 2: STAR 압축된 delta를 lambda 가중 평균 후 base에 합산

    Args:
        base_sd    : 베이스 모델 state dict
        expert_sds : 전문가 모델 state dict 리스트
        eta        : nuclear norm 유지 비율 (%, 기본 40). 낮을수록 더 많이 압축
        lambdas    : 전문가별 스케일 계수 (논문 기본: [1/N, ...] 균등 평균)
    """
    keys = get_numeric_keys(base_sd, expert_sds)
    N = len(expert_sds)

    # ── Step 1: 각 expert delta에 STAR 압축 ──
    star_deltas = []
    for i, expert_sd in enumerate(expert_sds):
        print(f"  [STAR] Expert {i}: eta={eta}")
        compressed = {}
        with torch.no_grad():
            for key in tqdm(keys, desc=f"    Expert {i} STAR SVD"):
                delta = expert_sd[key].float() - base_sd[key].float()
                compressed[key] = _star_compress_tensor(delta, eta)
        star_deltas.append(compressed)

    # ── Step 2: lambda 가중 평균 + base 합산 ──
    final_sd = {}
    with torch.no_grad():
        for key in tqdm(keys, desc="  STAR merge"):
            merged_delta = torch.zeros_like(base_sd[key].float())
            for i in range(N):
                merged_delta.add_(lambdas[i] * star_deltas[i][key])
            final_sd[key] = base_sd[key].float() + merged_delta

    for key in base_sd:
        if key not in final_sd:
            final_sd[key] = base_sd[key]

    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 5. CART (Centered And Rank-Truncated, EMNLP 2024)
# ──────────────────────────────────────────────────────────────────────────────

def _cart_lowrank_tensor(delta: torch.Tensor, rank_ratio: float) -> torch.Tensor:
    """2D tensor에 low-rank approximation 적용 (CART SVD truncation)."""
    u, s, vt = torch.linalg.svd(delta, full_matrices=False)
    min_dim = s.shape[0]
    rank = max(1, int(rank_ratio * min_dim))
    rank = min(rank, min_dim)
    return u[:, :rank] @ torch.diag(s[:rank]) @ vt[:rank, :]


def run_cart(
    base_sd: dict, expert_sds: List[dict],
    prior: float, rank_ratio: float,
) -> dict:
    """
    CART: Centered And Rank-Truncated merge (EMNLP 2024).

    1. theta_avg  = base + (1/N) · Σ delta_i          (weight average)
    2. c_delta_i  = expert_i - theta_avg               (centering)
    3. low-rank approx on 2D layers via SVD truncation  (rank = rank_ratio · min_dim)
    4. merged     = theta_avg + prior · Σ trunc(c_delta_i)

    논문 권장 파라미터 (ViT-B/32, 8/14/20 tasks):
      prior      : 2.0 / 1.5 / 1.9
      rank_ratio : 0.12 / 0.16 / 0.32
    """
    keys = get_numeric_keys(base_sd, expert_sds)
    N = len(expert_sds)

    # ── Step 1: theta_avg ──
    print(f"  [CART] Step 1: computing weight average (theta_avg)")
    theta_avg = {}
    with torch.no_grad():
        for key in keys:
            base = base_sd[key].float()
            delta_sum = torch.zeros_like(base)
            for expert_sd in expert_sds:
                delta_sum.add_(expert_sd[key].float() - base)
            theta_avg[key] = base + delta_sum / N

    # ── Step 2 & 3: centered deltas + low-rank approximation ──
    print(f"  [CART] Step 2-3: centered deltas + SVD low-rank (rank_ratio={rank_ratio})")
    lowrank_sum = {}
    with torch.no_grad():
        for key in tqdm(keys, desc="  CART SVD"):
            apply_svd = (base_sd[key].dim() == 2)
            centered_sum = torch.zeros_like(theta_avg[key])
            for expert_sd in expert_sds:
                c_delta = expert_sd[key].float() - theta_avg[key]
                if apply_svd:
                    c_delta = _cart_lowrank_tensor(c_delta, rank_ratio)
                centered_sum.add_(c_delta)
            lowrank_sum[key] = centered_sum

    # ── Step 4: final merge ──
    final_sd = {}
    with torch.no_grad():
        for key in keys:
            final_sd[key] = theta_avg[key] + prior * lowrank_sum[key]

    for key in base_sd:
        if key not in final_sd:
            final_sd[key] = base_sd[key]

    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 6. TSV (Task Singular Vectors)  [formerly section 3]
# ──────────────────────────────────────────────────────────────────────────────

def run_tsv(
    base_sd: dict, expert_sds: List[dict],
    alpha: float, k: Optional[int], sv_reduction: Optional[float],
    device: str,
) -> dict:
    """
    SVD 기반 직교화 후 재구성.
    2D 레이어: block-diagonal SVD → polar 직교화 → 재구성
    1D 레이어: rolling mean
    """
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
                # 2D: SVD block-diagonal
                k_per = None
                sum_u = sum_s = sum_v = None

                for i, tv in enumerate(tv_list):
                    U, S, Vh = torch.linalg.svd(tv, full_matrices=False)
                    if i == 0:
                        min_dim = S.shape[0]
                        k_per = min(k, min_dim) if k else max(1, int(min_dim * sv_reduction))
                        total_k = N * k_per
                        if total_k > min_dim:
                            k_per = max(1, min_dim // N)
                        M, D = tv.shape
                        sum_u = torch.zeros(M, min_dim, device=device)
                        sum_s = torch.zeros(min_dim, device=device)
                        sum_v = torch.zeros(min_dim, D, device=device)

                    sum_u[:, i * k_per:(i + 1) * k_per] = U[:, :k_per]
                    sum_s[i * k_per:(i + 1) * k_per] = S[:k_per]
                    sum_v[i * k_per:(i + 1) * k_per, :] = Vh[:k_per, :]
                    del U, S, Vh

                u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                merged_tv[key] = torch.linalg.multi_dot((
                    u_u, v_u, torch.diag(sum_s), u_v, v_v
                )).cpu()
                del sum_u, sum_s, sum_v, u_u, v_u, u_v, v_v
            else:
                # 1D: rolling mean
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


# ──────────────────────────────────────────────────────────────────────────────
# 7. Fisher Precision-Weighted Merge
# ──────────────────────────────────────────────────────────────────────────────

def _compute_empirical_fisher(
    model_path: str, device: str, calib_samples: int, calib_seqlen: int, seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """경험적 Fisher 대각선 추정 (gradient^2 평균)."""
    print(f"\n  Computing Fisher: {model_path}")
    torch.manual_seed(seed)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).eval()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.gradient_checkpointing_enable()
    vocab_size = model.config.vocab_size

    # device_map="auto" 시 첫 번째 파라미터의 디바이스를 input 디바이스로 사용
    first_device = next(model.parameters()).device

    params = [(n, p) for n, p in model.named_parameters() if p.data.is_floating_point()]
    for _, p in params:
        p.requires_grad_(True)

    fisher = {n: torch.zeros_like(p, dtype=torch.float32, device="cpu") for n, p in params}

    for s in tqdm(range(calib_samples), desc="  Fisher estimation"):
        input_ids = torch.randint(0, vocab_size, (1, calib_seqlen), device=first_device)
        model.zero_grad()
        logits = model(input_ids=input_ids).logits
        log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        seq_lp = log_probs.gather(2, input_ids[:, 1:].unsqueeze(2)).sum()
        seq_lp.backward()
        for n, p in params:
            if p.grad is not None:
                fisher[n].add_(p.grad.detach().float().cpu().pow(2))
        del logits, log_probs, seq_lp
        torch.cuda.empty_cache()

    for n in fisher:
        fisher[n].div_(calib_samples)

    del model, params
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return fisher


def run_fisher(
    base_sd: dict, expert_sds: List[dict], expert_paths: List[str],
    lambdas: List[float], epsilon: float,
    fisher_files: Optional[List[str]], calib_samples: int, calib_seqlen: int,
    fisher_device: str, save_fisher: bool, save_dir: str,
) -> dict:
    """
    Fisher precision-weighted merge (delta-space):
    θ* = base + Σ(λ_i·F_i·Δ_i) / (Σ(λ_i·F_i) + ε)
    """
    # Fisher 로드 or 계산
    fishers = []
    if fisher_files:
        for fp in fisher_files:
            print(f"  Loading Fisher: {fp}")
            fishers.append(torch.load(fp, map_location="cpu"))
    else:
        fisher_save_dir = os.path.join(save_dir, "fisher_diagonals") if save_fisher else None
        if fisher_save_dir:
            os.makedirs(fisher_save_dir, exist_ok=True)
        for i, path in enumerate(expert_paths):
            # 이전 모델 잔여 메모리 완전 해제
            gc.collect()
            torch.cuda.empty_cache()
            f = _compute_empirical_fisher(path, fisher_device, calib_samples, calib_seqlen, 42 + i)
            fishers.append(f)
            if fisher_save_dir:
                torch.save(f, os.path.join(fisher_save_dir, f"fisher_{i}.pt"))

    # 공통 키
    common = set(base_sd.keys())
    for sd in expert_sds:
        common &= set(sd.keys())
    for f in fishers:
        common &= set(f.keys())
    keys = sorted(k for k in common if base_sd[k].dtype not in (torch.int64, torch.uint8))

    N = len(expert_sds)
    final_sd = {}
    zero_count = 0

    with torch.no_grad():
        for key in tqdm(keys, desc="Fisher Merge"):
            base = base_sd[key].float()
            num = torch.zeros_like(base)
            den = torch.zeros_like(base)
            for i in range(N):
                delta = expert_sds[i][key].float() - base
                prec = lambdas[i] * fishers[i][key].float()
                num.add_(prec * delta)
                den.add_(prec)
            zero_mask = den <= epsilon
            merged = base + num / (den + epsilon)
            merged = torch.where(zero_mask, base, merged)
            final_sd[key] = merged
            zero_count += int(zero_mask.sum().item())

    for key in base_sd:
        if key not in final_sd:
            final_sd[key] = base_sd[key]

    print(f"  Zero-precision params: {zero_count} → base preserved")
    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 5. Whitened (Polar) Merge — 공통 핵심 로직
# ──────────────────────────────────────────────────────────────────────────────

def polar_factor(X: torch.Tensor) -> torch.Tensor:
    """가장 가까운 직교행렬: SVD(X)의 U @ Vh."""
    U, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return U @ Vh


def _whitened_merge_core(
    base_sd: dict,
    expert_sds: List[dict],
    k_list: List[int],
    mode: str,  # "scaled" | "noscale" | "per_tv" | "capped"
) -> tuple:
    """
    Polar whitening 기반 머징의 핵심 로직.

    mode 별 동작:
      scaled   : base + α_wht · Σ τ_wht_i  (per-layer α, energy 보존)
      noscale  : base + Σ τ_wht_i            (α=1, ablation)
      per_tv   : base + Σ α_i · τ_wht_i     (per-expert energy 개별 복원)
      capped   : base + α_global · Σ min(α_i, α_wht)·τ_wht_i

    Returns:
        ({k: merged_sd}, {k: layer_stats})
    """
    N = len(expert_sds)
    k_list = sorted(k_list)
    max_k = max(k_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    common = set(base_sd.keys())
    for sd in expert_sds:
        common &= set(sd.keys())

    keys_2d = sorted(
        key for key in common
        if base_sd[key].ndim == 2
        and base_sd[key].dtype in (torch.float32, torch.float16, torch.bfloat16)
    )
    keys_other = [k for k in base_sd if k not in set(keys_2d)]

    merged_per_k = {k: {} for k in k_list}
    stats_per_k = {k: {} for k in k_list}

    with torch.no_grad():
        for key in tqdm(keys_2d, desc=f"Whitened [{mode}] 2D"):
            W_base = base_sd[key].float().to(device)
            min_dim = min(W_base.shape)

            task_vecs = [sd[key].float().to(device) - W_base for sd in expert_sds]
            tv_norms_sq = [tau.norm().item() ** 2 for tau in task_vecs]
            D_sq = sum(tv_norms_sq)

            # SVD (최대 max_k까지)
            max_k_eff = min(max_k, min_dim // N)
            if max_k_eff == 0:
                # min_dim < N인 극소 레이어: whitening 불가 → simple mean fallback
                mean_tv = sum(task_vecs) / N
                for k_val in k_list:
                    merged_per_k[k_val][key] = (W_base + mean_tv).cpu()
                    stats_per_k[k_val][key] = {"k_eff": 0, "fallback": "mean"}
                continue

            all_U, all_S, all_V = [], [], []
            for tau in task_vecs:
                U, S, Vh = torch.linalg.svd(tau, full_matrices=False)
                all_U.append(U[:, :max_k_eff])
                all_S.append(S[:max_k_eff])
                all_V.append(Vh[:max_k_eff, :].T)  # (D, max_k_eff)

            for k_val in k_list:
                k_eff = min(k_val, max_k_eff)

                Us = [u[:, :k_eff] for u in all_U]
                Ss = [s[:k_eff]    for s in all_S]
                Vs = [v[:, :k_eff] for v in all_V]

                # Polar factor
                U_hat = polar_factor(torch.cat(Us, dim=1))
                V_hat = polar_factor(torch.cat(Vs, dim=1))

                wht_norms_sq = [(s ** 2).sum().item() for s in Ss]
                wht_sq_total = sum(wht_norms_sq)

                if mode == "scaled":
                    # Σ τ_wht_i 먼저 계산
                    tau_wht = torch.zeros_like(W_base)
                    for i in range(N):
                        s, e = i * k_eff, (i + 1) * k_eff
                        tau_wht += U_hat[:, s:e] @ torch.diag(Ss[i]) @ V_hat[:, s:e].T
                    wht_sq = tau_wht.norm().item() ** 2
                    alpha_wht = math.sqrt(D_sq / (wht_sq + 1e-12)) if wht_sq > 1e-12 else 1.0
                    merged_per_k[k_val][key] = (W_base + alpha_wht * tau_wht).cpu()
                    stats_per_k[k_val][key] = {"k_eff": k_eff, "alpha_wht": alpha_wht, "D_sq": D_sq}

                elif mode == "noscale":
                    tau_wht = torch.zeros_like(W_base)
                    for i in range(N):
                        s, e = i * k_eff, (i + 1) * k_eff
                        tau_wht += U_hat[:, s:e] @ torch.diag(Ss[i]) @ V_hat[:, s:e].T
                    merged_per_k[k_val][key] = (W_base + tau_wht).cpu()
                    stats_per_k[k_val][key] = {"k_eff": k_eff, "D_sq": D_sq}

                elif mode == "per_tv":
                    tau_sum = torch.zeros_like(W_base)
                    expert_stats = []
                    for i in range(N):
                        s, e = i * k_eff, (i + 1) * k_eff
                        tau_wht_i = U_hat[:, s:e] @ torch.diag(Ss[i]) @ V_hat[:, s:e].T
                        wht_i_sq = tau_wht_i.norm().item() ** 2
                        alpha_i = math.sqrt(tv_norms_sq[i] / (wht_i_sq + 1e-12)) if wht_i_sq > 1e-12 else 1.0
                        tau_sum += alpha_i * tau_wht_i
                        expert_stats.append({"orig_norm_sq": tv_norms_sq[i], "alpha_i": alpha_i})
                    merged_per_k[k_val][key] = (W_base + tau_sum).cpu()
                    stats_per_k[k_val][key] = {"k_eff": k_eff, "D_sq": D_sq, "experts": expert_stats}

                elif mode == "capped":
                    alpha_wht = math.sqrt(D_sq / (wht_sq_total + 1e-12)) if wht_sq_total > 1e-12 else 1.0
                    tau_capped = torch.zeros_like(W_base)
                    expert_stats = []
                    for i in range(N):
                        s, e = i * k_eff, (i + 1) * k_eff
                        tau_wht_i = U_hat[:, s:e] @ torch.diag(Ss[i]) @ V_hat[:, s:e].T
                        alpha_i_raw = math.sqrt(tv_norms_sq[i] / (wht_norms_sq[i] + 1e-12)) if wht_norms_sq[i] > 1e-12 else 1.0
                        alpha_i = min(alpha_i_raw, alpha_wht)
                        tau_capped += alpha_i * tau_wht_i
                        expert_stats.append({"alpha_i_raw": alpha_i_raw, "alpha_i": alpha_i, "capped": alpha_i_raw > alpha_wht})
                    capped_sq = tau_capped.norm().item() ** 2
                    alpha_global = math.sqrt(D_sq / (capped_sq + 1e-12)) if capped_sq > 1e-12 else 1.0
                    merged_per_k[k_val][key] = (W_base + alpha_global * tau_capped).cpu()
                    stats_per_k[k_val][key] = {"k_eff": k_eff, "alpha_wht": alpha_wht, "alpha_global": alpha_global, "D_sq": D_sq, "experts": expert_stats}

    # 1D / 비-numeric 레이어: task arithmetic 평균 (non-float는 base 그대로)
    for key in keys_other:
        if base_sd[key].is_floating_point():
            W_base = base_sd[key].float()
            tvs = [sd[key].float() - W_base for sd in expert_sds if key in sd
                   if sd[key].is_floating_point()]
            val = (W_base + sum(tvs) / len(tvs)) if tvs else W_base.clone()
        else:
            val = base_sd[key].clone()
        for k_val in k_list:
            merged_per_k[k_val][key] = val.clone()

    return merged_per_k, stats_per_k


# ──────────────────────────────────────────────────────────────────────────────
# 5.5  RMT-based Polar Whitening (Gavish-Donoho automatic k + optimal shrinkage)
# ──────────────────────────────────────────────────────────────────────────────

def _optimal_shrinkage(S: torch.Tensor, beta: float, sigma: float) -> torch.Tensor:
    """
    Donoho & Gavish (2017) optimal shrinkage for Frobenius loss.

    주어진 singular value y에 대해 de-biased estimator:
        η(y) = (1/y) * sqrt((y² - λ₊)(y² - λ₋))₊

    여기서 λ₊ = σ²(1+√β)², λ₋ = σ²(1-√β)²

    이건 truncated SVD의 singular value가 가지는 upward bias를 보정합니다.
    """
    lam_plus = sigma**2 * (1 + math.sqrt(beta))**2
    lam_minus = sigma**2 * (1 - math.sqrt(beta))**2

    y_sq = S ** 2
    inner = (y_sq - lam_plus) * (y_sq - lam_minus)
    inner = torch.clamp(inner, min=0.0)
    shrunk = torch.sqrt(inner) / S.clamp(min=1e-12)
    return shrunk


def run_rmt_merge(
    base_sd: dict,
    expert_sds: List[dict],
    mode: str = "per_tv",  # "scaled" | "per_tv"
    shrinkage: str = "energy",  # "energy" | "optimal" | "none"
    k_max_cap: int = 1024,
    k_min_floor: int = 0,
) -> tuple:
    """
    RMT 기반 Polar Whitening 머징.

    기존 whitened/per_tv_renorm과 동일한 pipeline이지만:
    - k를 Gavish-Donoho optimal threshold로 자동 결정 (per expert, per layer)
    - 선택적으로 Donoho-Gavish (2017) optimal shrinkage로 singular value de-bias

    Args:
        base_sd: base model state dict
        expert_sds: list of expert state dicts
        mode: "scaled" (per-layer α) or "per_tv" (per-expert α)
        shrinkage: bias correction 방식
            "energy"  : 기존 PER 방식 — Frobenius norm 보존 α rescaling
            "optimal" : Donoho-Gavish 2017 optimal shrinkage (singular value별 de-bias)
            "none"    : bias correction 없음 (ablation)
        k_max_cap: k 상한 (메모리 제한용)

    Returns:
        (merged_sd, stats) where stats[key] has per-layer diagnostics
    """
    N = len(expert_sds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    common = set(base_sd.keys())
    for sd in expert_sds:
        common &= set(sd.keys())

    keys_2d = sorted(
        key for key in common
        if base_sd[key].ndim == 2
        and base_sd[key].dtype in (torch.float32, torch.float16, torch.bfloat16)
    )
    keys_other = [k for k in base_sd if k not in set(keys_2d)]

    merged_sd = {}
    layer_stats = {}

    # RMT k 분포 통계 수집
    all_ks = []

    with torch.no_grad():
        for key in tqdm(keys_2d, desc=f"RMT Merge [{mode}, shrink={shrinkage}]"):
            W_base = base_sd[key].float().to(device)
            m, n = W_base.shape
            min_dim = min(m, n)

            task_vecs = [sd[key].float().to(device) - W_base for sd in expert_sds]
            tv_norms_sq = [tau.norm().item() ** 2 for tau in task_vecs]
            D_sq = sum(tv_norms_sq)

            # ── 각 expert의 SVD + RMT threshold로 k 결정 ──
            if min_dim < N:
                # min_dim < N인 극소 레이어: polar whitening 불가 → simple mean fallback
                mean_tv = sum(task_vecs) / N
                merged_sd[key] = (W_base + mean_tv).cpu()
                layer_stats[key] = {"fallback": "mean", "k_per_expert": [0]*N}
                continue

            svd_results = []
            k_per_expert = []
            beta = min(m, n) / max(m, n)

            for i, tau in enumerate(task_vecs):
                U, S, Vh = torch.linalg.svd(tau, full_matrices=False)

                # Gavish-Donoho threshold로 이 expert의 signal rank 결정
                k_rmt = gavish_donoho_threshold(S, m, n)
                k_rmt = max(k_rmt, min(k_min_floor, min_dim))
                k_rmt = min(k_rmt, k_max_cap)
                k_per_expert.append(k_rmt)

                svd_results.append((U, S, Vh))

            # polar factor의 총 column 수 = sum(k_rmt_i) ≤ min_dim 제약
            total_k = sum(k_per_expert)
            if total_k > min_dim:
                # 비례적으로 축소: 각 k를 같은 비율로 줄임
                scale = min_dim / total_k
                k_per_expert = [max(1, int(k * scale)) for k in k_per_expert]
                # 반올림 오차로 여전히 초과할 수 있으므로 보정
                while sum(k_per_expert) > min_dim:
                    idx = max(range(N), key=lambda j: k_per_expert[j])
                    k_per_expert[idx] -= 1

            Us, Ss, Vs = [], [], []
            signal_norms_sq = []  # 각 expert의 signal-only energy (noise 제외)

            for i, (U, S, Vh) in enumerate(svd_results):
                k_i = k_per_expert[i]
                u_k = U[:, :k_i]
                s_k = S[:k_i].clone()
                v_k = Vh[:k_i, :].T  # (n, k_i)

                # signal energy = threshold 이상 singular value의 에너지
                sig_energy = (s_k ** 2).sum().item()
                signal_norms_sq.append(sig_energy)

                if shrinkage == "optimal":
                    # noise level 추정: median singular value
                    sigma_hat = S[len(S) // 2].item()
                    # signal singular value에 optimal shrinkage 적용
                    s_k = _optimal_shrinkage(s_k, beta, sigma_hat)

                Us.append(u_k)
                Ss.append(s_k)
                Vs.append(v_k)

            # Polar factor — 각 expert가 자기 k_rmt만큼만 기여
            U_hat = polar_factor(torch.cat(Us, dim=1))
            V_hat = polar_factor(torch.cat(Vs, dim=1))

            # signal energy 합
            S_sq = sum(signal_norms_sq)

            # expert별 column offset 계산 (k가 다르므로 누적합)
            offsets = [0]
            for k_i in k_per_expert:
                offsets.append(offsets[-1] + k_i)

            if mode == "scaled":
                tau_wht = torch.zeros_like(W_base)
                for i in range(N):
                    s, e = offsets[i], offsets[i + 1]
                    tau_wht += U_hat[:, s:e] @ torch.diag(Ss[i]) @ V_hat[:, s:e].T

                if shrinkage == "energy":
                    # 전체 energy 기준 복원: denoised 방향을 원래 update 강도로
                    wht_sq = tau_wht.norm().item() ** 2
                    alpha_wht = math.sqrt(D_sq / (wht_sq + 1e-12)) if wht_sq > 1e-12 else 1.0
                    merged_sd[key] = (W_base + alpha_wht * tau_wht).cpu()
                    layer_stats[key] = {
                        "k_per_expert": k_per_expert,
                        "alpha_wht": alpha_wht, "D_sq": D_sq, "S_sq": S_sq,
                    }
                else:
                    merged_sd[key] = (W_base + tau_wht).cpu()
                    layer_stats[key] = {
                        "k_per_expert": k_per_expert,
                        "D_sq": D_sq, "S_sq": S_sq,
                    }

            elif mode == "per_tv":
                tau_sum = torch.zeros_like(W_base)
                expert_stats = []
                for i in range(N):
                    s, e = offsets[i], offsets[i + 1]
                    tau_wht_i = U_hat[:, s:e] @ torch.diag(Ss[i]) @ V_hat[:, s:e].T

                    if shrinkage == "energy":
                        # 전체 energy 기준 복원: denoised 방향을 원래 update 강도로
                        wht_i_sq = tau_wht_i.norm().item() ** 2
                        alpha_i = math.sqrt(tv_norms_sq[i] / (wht_i_sq + 1e-12)) if wht_i_sq > 1e-12 else 1.0
                        tau_sum += alpha_i * tau_wht_i
                        expert_stats.append({
                            "k_rmt": k_per_expert[i],
                            "orig_norm_sq": tv_norms_sq[i],
                            "signal_norm_sq": signal_norms_sq[i],
                            "alpha_i": alpha_i,
                        })
                    else:
                        tau_sum += tau_wht_i
                        expert_stats.append({
                            "k_rmt": k_per_expert[i],
                            "orig_norm_sq": tv_norms_sq[i],
                            "signal_norm_sq": signal_norms_sq[i],
                        })

                merged_sd[key] = (W_base + tau_sum).cpu()
                layer_stats[key] = {
                    "k_per_expert": k_per_expert,
                    "D_sq": D_sq, "S_sq": S_sq, "experts": expert_stats,
                }

            all_ks.extend(k_per_expert)

    # 1D / non-float 레이어
    for key in keys_other:
        if base_sd[key].is_floating_point():
            W_base = base_sd[key].float()
            tvs = [sd[key].float() - W_base for sd in expert_sds if key in sd
                   if sd[key].is_floating_point()]
            val = (W_base + sum(tvs) / len(tvs)) if tvs else W_base.clone()
        else:
            val = base_sd[key].clone()
        merged_sd[key] = val

    # RMT k 분포 요약 출력
    if all_ks:
        ks_arr = np.array(all_ks)
        print(f"\n  [RMT k 분포] mean={ks_arr.mean():.1f}, median={np.median(ks_arr):.0f}, "
              f"min={ks_arr.min()}, max={ks_arr.max()}, std={ks_arr.std():.1f}")
        # component type별 통계
        comp_ks = defaultdict(list)
        for key, st in layer_stats.items():
            if "k_per_expert" in st:
                # key에서 component type 추출 (e.g., q_proj, k_proj, gate_proj 등)
                parts = key.split(".")
                comp_type = parts[-1] if parts[-1] != "weight" else parts[-2]
                comp_ks[comp_type].extend(st["k_per_expert"])
        print(f"  [Component별 평균 k]")
        for comp, ks in sorted(comp_ks.items()):
            ka = np.array(ks)
            print(f"    {comp:20s}: mean={ka.mean():7.1f}  median={np.median(ka):5.0f}  "
                  f"min={ka.min():4d}  max={ka.max():4d}")

    return merged_sd, layer_stats


# ──────────────────────────────────────────────────────────────────────────────
# 6. Energy-Direction Disentanglement (실험용 전체 변형 생성)
# ──────────────────────────────────────────────────────────────────────────────

def run_energy_direction(
    base_sd: dict, expert_sds: List[dict], k_list: List[int],
) -> tuple:
    """
    에너지 vs 방향 분리 실험. 한 번 실행으로 아래 변형 모두 생성:
      (a) direct_sum     : base + Σ τ_i
      (b) scaled_sum     : base + α·Σ τ_i (per-layer)
      (c) truncated_sum  : base + Σ τ_i_trunc (per k)
      (d) whitened       : base + α_wht·Σ τ_wht_i (per-layer, per k)
      (e) whitened_noscale: base + Σ τ_wht_i (per k)
      (f) global_whitened: base + α_global·Σ τ_wht_i (per k)

    Returns:
        (sd_direct, sd_scaled, {k: sd_trunc}, {k: sd_wht}, {k: sd_noscale}, {k: sd_gwht}, {k: stats})
    """
    N = len(expert_sds)
    k_list = sorted(k_list)
    max_k = max(k_list)

    common = set(base_sd.keys())
    for sd in expert_sds:
        common &= set(sd.keys())
    keys_2d = sorted(k for k in common if base_sd[k].ndim == 2 and base_sd[k].dtype in (torch.float32, torch.float16, torch.bfloat16))
    keys_other = [k for k in base_sd if k not in set(keys_2d)]

    sd_direct = {}
    sd_scaled = {}
    sd_trunc = {k: {} for k in k_list}
    sd_wht = {k: {} for k in k_list}
    sd_noscale = {k: {} for k in k_list}
    sd_gwht = {k: {} for k in k_list}
    stats = {k: {} for k in k_list}
    alpha_wht_list = {k: [] for k in k_list}

    with torch.no_grad():
        for key in tqdm(keys_2d, desc="Energy-Direction"):
            W_base = base_sd[key].float()
            min_dim = min(W_base.shape)
            task_vecs = [sd[key].float() - W_base for sd in expert_sds]

            D_sq = sum(t.norm().item() ** 2 for t in task_vecs)
            tau_dir = sum(task_vecs)
            dir_sq = tau_dir.norm().item() ** 2
            interference = dir_sq - D_sq
            alpha = math.sqrt(D_sq / (dir_sq + 1e-12)) if dir_sq > 1e-12 else 1.0

            sd_direct[key] = W_base + tau_dir
            sd_scaled[key] = W_base + alpha * tau_dir

            max_k_eff = min(max_k, min_dim // N)
            if max_k_eff == 0:
                # min_dim < N인 극소 레이어: whitening 불가 → simple mean fallback
                mean_val = W_base + sum(task_vecs) / N
                sd_direct[key] = W_base + sum(task_vecs)
                sd_scaled[key] = mean_val
                for k_val in k_list:
                    sd_trunc[k_val][key] = mean_val
                    sd_wht[k_val][key] = mean_val
                    sd_noscale[k_val][key] = mean_val
                    sd_gwht[k_val][key] = mean_val
                    stats[k_val][key] = {"fallback": "mean", "D_sq": D_sq}
                continue

            all_Us, all_Ss, all_Vs = [], [], []
            for tau in task_vecs:
                U, S, Vh = torch.linalg.svd(tau, full_matrices=False)
                all_Us.append(U[:, :max_k_eff])
                all_Ss.append(S[:max_k_eff])
                all_Vs.append(Vh[:max_k_eff, :].T)

            for k_val in k_list:
                k_eff = min(k_val, min_dim // N)
                Us = [u[:, :k_eff] for u in all_Us]
                Ss = [s[:k_eff]    for s in all_Ss]
                Vs = [v[:, :k_eff] for v in all_Vs]

                # (c) truncated sum
                tau_trunc = sum(Us[i] @ torch.diag(Ss[i]) @ Vs[i].T for i in range(N))
                sd_trunc[k_val][key] = W_base + tau_trunc

                # polar factors
                U_hat = polar_factor(torch.cat(Us, dim=1))
                V_hat = polar_factor(torch.cat(Vs, dim=1))

                tau_wht = torch.zeros_like(W_base)
                for i in range(N):
                    s, e = i * k_eff, (i + 1) * k_eff
                    tau_wht += U_hat[:, s:e] @ torch.diag(Ss[i]) @ V_hat[:, s:e].T

                wht_sq = tau_wht.norm().item() ** 2
                trunc_D_sq = sum(s.norm().item() ** 2 for s in Ss)
                alpha_wht = math.sqrt(D_sq / (wht_sq + 1e-12)) if wht_sq > 1e-12 else 1.0
                alpha_wht_list[k_val].append(alpha_wht)

                sd_wht[k_val][key] = W_base + alpha_wht * tau_wht
                sd_noscale[k_val][key] = W_base + tau_wht
                # sd_gwht는 나중에 global alpha 계산 후 채움

                stats[k_val][key] = {
                    "D_sq": D_sq, "dir_sq": dir_sq, "wht_sq_raw": wht_sq,
                    "alpha_wht": alpha_wht, "alpha": alpha,
                    "interference": interference,
                    "frac_interference": interference / (D_sq + 1e-12),
                    "trunc_D_sq": trunc_D_sq,
                    "constructive": interference > 0,
                }

    # global_whitened: 레이어별 alpha_wht 평균 사용
    for k_val in k_list:
        alpha_global = sum(alpha_wht_list[k_val]) / len(alpha_wht_list[k_val])
        stats[k_val]["global_alpha"] = alpha_global
        print(f"  k={k_val}: global α = {alpha_global:.6f}")
        for key in keys_2d:
            W_base = base_sd[key].float()
            tau_wht = sd_noscale[k_val][key] - W_base
            sd_gwht[k_val][key] = W_base + alpha_global * tau_wht

    # 1D / 비-2D 키 (non-float는 base 그대로)
    for key in keys_other:
        if base_sd[key].is_floating_point():
            W_base = base_sd[key].float()
            tvs = [sd[key].float() - W_base for sd in expert_sds if key in sd
                   if sd[key].is_floating_point()]
            val = (W_base + sum(tvs) / len(tvs)) if tvs else W_base.clone()
        else:
            val = base_sd[key].clone()
        sd_direct[key] = val
        sd_scaled[key] = val.clone()
        for k_val in k_list:
            sd_trunc[k_val][key] = val.clone()
            sd_wht[k_val][key] = val.clone()
            sd_noscale[k_val][key] = val.clone()
            sd_gwht[k_val][key] = val.clone()

    return sd_direct, sd_scaled, sd_trunc, sd_wht, sd_noscale, sd_gwht, stats


# ──────────────────────────────────────────────────────────────────────────────
# 6-g. CAT (Conflict-Aware Task merging)
#      출처: https://github.com/SunWenJu123/model-merging
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_c4_calibration(tokenizer, calib_samples: int, calib_seqlen: int, seed: int = 42):
    """
    C4 데이터셋에서 calibration 토큰 시퀀스를 준비.
    Returns: List[torch.Tensor] — 각 (1, calib_seqlen) shape의 input_ids
    """
    import random
    from datasets import load_dataset

    random.seed(seed)
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)

    # 긴 텍스트를 연결하여 calib_seqlen 단위로 자름
    all_tokens = []
    for sample in ds:
        tokens = tokenizer(sample["text"], return_tensors="pt", add_special_tokens=False).input_ids[0]
        all_tokens.append(tokens)
        total = sum(t.numel() for t in all_tokens)
        if total >= calib_samples * calib_seqlen * 2:
            break

    concat = torch.cat(all_tokens, dim=0)
    result = []
    for i in range(calib_samples):
        start = i * calib_seqlen
        if start + calib_seqlen > concat.numel():
            break
        result.append(concat[start : start + calib_seqlen].unsqueeze(0))

    return result


def _collect_activations_for_expert(
    base_model_path: str,
    task_vector_sd: Dict[str, torch.Tensor],
    calib_inputs: list,
    scaling_coef: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    전문가 모델(base + scaling_coef * task_vector)을 로드 → calibration 데이터로 forward pass →
    각 Linear/LayerNorm 레이어의 input activation을 수집하여 반환.

    원본(ViT + 이미지)의 hook 기반 activation 수집을 LLM용으로 이식.
    원본: model = tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)

    Args:
        calib_inputs: _prepare_c4_calibration() 반환값 (List[Tensor])
        scaling_coef: task vector 적용 계수 (CAT=args.scaling_coef, LOT=1.0)

    Returns:
        {param_name: input_features_tensor}  shape = (total_tokens, feat_dim)
        LayerNorm/RMSNorm의 경우: (output - bias) / weight  (de-normalized input)
    """
    # 1) 모델 로드 (base + scaling_coef * task_vector 적용)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float32, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    base_sd_local = model.state_dict()
    new_sd = {}
    for key in base_sd_local:
        if key in task_vector_sd:
            new_sd[key] = base_sd_local[key].float() + scaling_coef * task_vector_sd[key].float()
        else:
            new_sd[key] = base_sd_local[key]
    model.load_state_dict(new_sd, strict=True)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    del new_sd
    gc.collect()

    first_device = next(model.parameters()).device

    # 2) hook 대상 파라미터 → 모듈 매핑
    accumulated_features = defaultdict(list)
    accumulated_activations = defaultdict(list)

    def hook_fn(param_name, module, inp, out):
        x = inp[0].detach().float().cpu()
        accumulated_features[param_name].append(x)
        y = out.detach().float().cpu() if not isinstance(out, tuple) else out[0].detach().float().cpu()
        accumulated_activations[param_name].append(y)

    hooks = []
    hooked_params = set()
    param_to_module = {}

    for pname, param in model.named_parameters():
        if not param.data.is_floating_point():
            continue
        parts = pname.split(".")
        if parts[-1] != "weight":
            continue

        module_path = ".".join(parts[:-1])
        if not module_path:
            continue

        module = model.get_submodule(module_path)

        # Linear (2D weight)
        if isinstance(module, torch.nn.Linear):
            hook = module.register_forward_hook(partial(hook_fn, pname))
            hooks.append(hook)
            hooked_params.add(pname)
            param_to_module[pname] = module
        # LayerNorm / RMSNorm (1D weight)
        elif isinstance(module, torch.nn.LayerNorm) or \
             ("norm" in module_path.lower() and param.ndim == 1):
            hook = module.register_forward_hook(partial(hook_fn, pname))
            hooks.append(hook)
            hooked_params.add(pname)
            param_to_module[pname] = module

    # 3) calibration forward pass (C4 실제 텍스트)
    for input_ids in calib_inputs:
        input_ids = input_ids.to(first_device)
        with torch.no_grad():
            model(input_ids=input_ids)
        del input_ids
        torch.cuda.empty_cache()

    # 4) hook 제거
    for h in hooks:
        h.remove()

    # 5) activation 후처리
    feat_dict = {}
    sd = model.state_dict()
    for pname in hooked_params:
        module = param_to_module[pname]
        input_list = accumulated_features.get(pname)
        output_list = accumulated_activations.get(pname)

        if input_list is None or len(input_list) == 0:
            continue

        if isinstance(module, torch.nn.LayerNorm) or \
           ("norm" in pname.lower() and sd[pname].ndim == 1):
            # LayerNorm/RMSNorm: de-normalized input = (output - bias) / weight
            output_cat = torch.cat(output_list, dim=0).view(-1, output_list[0].shape[-1])
            weight = sd[pname].float().cpu()
            bias_name = pname.rsplit(".", 1)[0] + ".bias"
            bias = sd.get(bias_name)
            if bias is not None:
                feat_dict[pname] = (output_cat - bias.float().cpu()) / (weight + 1e-12)
            else:
                # RMSNorm — bias 없음
                feat_dict[pname] = output_cat / (weight + 1e-12)
        else:
            # Linear: flatten input features → (total_tokens, f_in)
            feats = [f.view(-1, f.shape[-1]) for f in input_list]
            feat_dict[pname] = torch.cat(feats, dim=0)

    del model, accumulated_features, accumulated_activations, sd
    gc.collect()
    torch.cuda.empty_cache()

    return feat_dict


def run_cat(
    base_sd: dict, expert_sds: List[dict],
    expert_paths: List[str], base_model_path: str,
    scaling_coef: float,
    calib_samples: int, calib_seqlen: int,
    cat_ratio: float, cat_lambd: float,
    cat_ratio_ln: int, cat_lambd_ln: float,
    cat_ratio_bias: int,
    device: str = "cuda:0",
) -> dict:
    """
    CAT (Conflict-Aware Task merging) — LLM 버전.
    출처: SunWenJu123/model-merging  (cat_merging_utils.py 완전 이식)

    알고리즘:
      1. 각 전문가별 activation 수집 (C4 calibration + scaling_coef 적용 모델)
      2. 2D weight (Linear): conflict basis 계산 → project-out (norm 보존)
      3. 1D weight (LayerNorm): conflict가 큰 차원 마스킹 (norm 보존)
      4. bias/1D params: conflict가 큰 차원 마스킹
      5. 수정된 task vector들을 합산: base + scaling_coef * Σ τ_i
    """
    N = len(expert_sds)
    import numpy as np

    # task vectors 계산
    task_vectors = []
    for sd in expert_sds:
        tv = {}
        for key in base_sd:
            if key in sd and base_sd[key].is_floating_point():
                tv[key] = sd[key].float() - base_sd[key].float()
            else:
                tv[key] = torch.zeros_like(base_sd[key].float()) if base_sd[key].is_floating_point() else None
        task_vectors.append(tv)

    # result_tvs: 이것이 수정 대상 (원본의 result_task_vectors)
    result_tvs = [
        {k: v.clone() if v is not None else None for k, v in tv.items()}
        for tv in task_vectors
    ]

    # C4 calibration 데이터 준비
    print("  [CAT] Preparing C4 calibration data...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    calib_inputs = _prepare_c4_calibration(tokenizer, calib_samples, calib_seqlen, seed=42)
    print(f"    {len(calib_inputs)} sequences × {calib_seqlen} tokens")

    # activation 수집
    # 원본: model = tv.apply_to(pretrained_checkpoint, scaling_coef=args.scaling_coef_)
    # → base + scaling_coef * task_vector 로 모델 구성
    print("  [CAT] Collecting activations for each expert...")
    feat_arr = []
    for i, path in enumerate(expert_paths):
        print(f"    Expert {i}: {path}")
        feat = _collect_activations_for_expert(
            base_model_path, task_vectors[i],
            calib_inputs, scaling_coef=scaling_coef,
        )
        feat_arr.append(feat)
        gc.collect()
        torch.cuda.empty_cache()

    # ── (1) 2D weight: conflict basis 제거 ──
    # 원본: if args.ratio != 0:
    basis_arr = []
    if cat_ratio != 0:
        print(f"  [CAT] Removing conflict basis (ratio={cat_ratio}, λ={cat_lambd})...")
        keys_2d = sorted(
            k for k in base_sd
            if base_sd[k].ndim == 2 and base_sd[k].is_floating_point()
            and all(k in sd for sd in expert_sds)
            and all(k in feat_arr[t] for t in range(N))
        )

        for tv_id in range(N):
            basis = {}
            for key in tqdm(keys_2d, desc=f"    Basis (expert {tv_id})", leave=False):
                feat_i = feat_arr[tv_id].get(key)
                if feat_i is None:
                    basis[key] = None
                    continue

                # 원본: xtx_i = torch.t(feat_i).mm(feat_i)
                xtx_i = feat_i.T @ feat_i   # (f_in, f_in)

                G = None
                for tvj_id in range(N):
                    if tvj_id == tv_id:
                        continue
                    feat_j = feat_arr[tvj_id].get(key)
                    if feat_j is None:
                        continue
                    # 원본: xtx = xtx_i - args.lambd * torch.t(feat_j).mm(feat_j)
                    xtx = xtx_i - cat_lambd * (feat_j.T @ feat_j)
                    # 원본: w_j = tv_j.vector[name]
                    w_j = result_tvs[tvj_id][key]                   # (f_out, f_in)
                    # 원본: G_ = w_j.mm(xtx).mm(torch.t(w_j))
                    G_ = w_j @ xtx @ w_j.T                         # (f_out, f_out)
                    G = G_ if G is None else G + G_

                if G is None:
                    basis[key] = None
                    continue

                G = (G + G.T) / 2
                eigenvalues, eigenvectors = torch.linalg.eigh(G)

                if (eigenvalues <= 0).all():
                    basis[key] = None
                else:
                    positive_mask = eigenvalues > 0
                    positive_eigenvalues = eigenvalues[positive_mask]
                    positive_eigenvectors = eigenvectors[:, positive_mask]

                    if cat_ratio == -1.0:
                        # 원본: basis_ = positive_eigenvectors
                        basis[key] = positive_eigenvectors
                    elif cat_ratio >= 1:
                        # 원본: r = args.ratio; basis_ = eigenvectors[:, 0:r]
                        r = int(cat_ratio)
                        basis[key] = eigenvectors[:, :r]
                    else:
                        # 원본: val_ratio = ...; r = np.sum(np.cumsum(val_ratio) < args.ratio)
                        val_total = (positive_eigenvalues ** 2).sum()
                        val_ratio = (positive_eigenvalues ** 2) / val_total
                        r = int(np.sum(np.cumsum(val_ratio.cpu().numpy()) < cat_ratio))
                        basis[key] = eigenvectors[:, :r]

            basis_arr.append(basis)

        # 원본 루프 순서: for tvj_id → for tv_id (basis) → project out
        for tvj_id in range(N):
            for tv_id in range(N):
                if tv_id == tvj_id:
                    continue
                basis = basis_arr[tv_id]
                for key in keys_2d:
                    b = basis.get(key)
                    if b is None:
                        continue
                    # 원본: bbt_ = b.mm(torch.t(b))
                    bbt = b @ b.T                                   # (f_out, f_out)
                    vec = result_tvs[tvj_id][key]                   # (f_out, f_in)
                    # 원본: vector_orthogonal = (vector_j - torch.mm(bbt_, vector_j))
                    vec_orth = vec - bbt @ vec
                    norm_orig = vec.norm()
                    norm_orth = vec_orth.norm()
                    # 원본: tv.vector[name] = vector_orthogonal * (norm_original / norm_remaining)
                    if norm_orth > 0:
                        result_tvs[tvj_id][key] = vec_orth * (norm_orig / norm_orth)

    # ── (2) LayerNorm conflict masking ──
    # 원본: if args.ratio_ln != 0:
    if cat_ratio_ln != 0:
        print(f"  [CAT] LayerNorm conflict masking (ratio_ln={cat_ratio_ln}, λ_ln={cat_lambd_ln})...")
        keys_ln = sorted(
            k for k in base_sd
            if base_sd[k].ndim == 1 and base_sd[k].is_floating_point()
            and "norm" in k.lower() and k.endswith(".weight")
            and all(k in sd for sd in expert_sds)
            and all(k in feat_arr[t] for t in range(N))
        )

        for key in keys_ln:
            # 원본: if result_task_vectors[0].vector[name] is None: continue
            if result_tvs[0][key] is None:
                continue
            if feat_arr[0].get(key) is None:
                continue

            masks = []
            for tv_id in range(N):
                # 원본: out = 0
                out = torch.zeros_like(base_sd[key].float())
                for tvj_id in range(N):
                    if tv_id == tvj_id:
                        continue
                    feat_i = feat_arr[tv_id].get(key)
                    feat_j = feat_arr[tvj_id].get(key)
                    tv_j_vec = result_tvs[tvj_id][key]
                    if feat_i is None or feat_j is None or tv_j_vec is None:
                        continue
                    # 원본: out += mean((feat[tv_id] * tv_j.vector)**2, dim=0) - lambd_ln * mean((feat[tvj_id] * tv_j.vector)**2, dim=0)
                    out = out + torch.mean((feat_i * tv_j_vec) ** 2, dim=0) \
                              - cat_lambd_ln * torch.mean((feat_j * tv_j_vec) ** 2, dim=0)

                dim_size = out.numel()
                topk_k = min(cat_ratio_ln, dim_size)
                # 원본: topk_indices = torch.topk(out, args.ratio_ln, largest=True).indices
                topk_indices = torch.topk(out, topk_k, largest=True).indices
                # 원본: mask = torch.ones_like(out, dtype=torch.int); mask[topk_indices] = 0
                mask = torch.ones(dim_size, dtype=torch.float32)
                mask[topk_indices] = 0.0
                masks.append(mask)

            # 원본 루프: for tv_id → for tvj_id → mask & rescale
            for tv_id in range(N):
                for tvj_id in range(N):
                    if tv_id == tvj_id:
                        continue
                    mask = masks[tv_id]
                    vec = result_tvs[tvj_id][key]
                    if vec is None:
                        continue
                    orig_norm = vec.norm()
                    masked = vec * mask
                    masked_norm = masked.norm()
                    # 원본: if masked_norm > 0: ... else: tv_j.vector[name] = masked_vector
                    if masked_norm > 0:
                        result_tvs[tvj_id][key] = masked * (orig_norm / masked_norm)
                    else:
                        result_tvs[tvj_id][key] = masked

    # ── (3) bias/1D parameter conflict masking ──
    # 원본: if args.ratio_bias != 0:
    if cat_ratio_bias != 0:
        print(f"  [CAT] Bias/1D conflict masking (ratio_bias={cat_ratio_bias})...")
        # 대상: LN weight 제외, bias 또는 1D 파라미터
        keys_bias = sorted(
            k for k in base_sd
            if base_sd[k].is_floating_point()
            and all(k in sd for sd in expert_sds)
            and result_tvs[0].get(k) is not None
            and (base_sd[k].ndim == 1 or "bias" in k)
            and not ("norm" in k.lower() and k.endswith(".weight"))
        )

        for key in keys_bias:
            params = [tv[key] for tv in result_tvs]
            stacked = torch.stack(params)
            masks = torch.ones_like(stacked)

            for i in range(N):
                # 원본: other_tasks_squared_sum = sum of task_i**2 for each j!=i
                # (원본 그대로 재현 — task_i 자신의 제곱을 (N-1)번 누적)
                other_tasks_squared_sum = torch.zeros_like(result_tvs[i][key])
                for j in range(N):
                    if i != j:
                        other_tasks_squared_sum = other_tasks_squared_sum + result_tvs[i][key] ** 2

                flattened = other_tasks_squared_sum.view(-1)
                num_elements = flattened.numel()
                if cat_ratio_bias >= 1:
                    k_val = num_elements - int(cat_ratio_bias)
                else:
                    k_val = int(num_elements * (1 - cat_ratio_bias))
                k_val = max(1, min(k_val, num_elements))
                threshold = torch.kthvalue(flattened, k_val).values
                mask_k = (other_tasks_squared_sum < threshold).float()

                for j in range(N):
                    if i != j:
                        masks[j] *= mask_k

            for j in range(N):
                result_tvs[j][key] = result_tvs[j][key] * masks[j]

    # ── (4) 최종 합산: base + scaling_coef * Σ τ_i ──
    print("  [CAT] Summing modified task vectors...")
    final_sd = {}
    for key in base_sd:
        if base_sd[key].is_floating_point():
            merged = base_sd[key].float().clone()
            for tv in result_tvs:
                if tv[key] is not None:
                    merged += scaling_coef * tv[key]
            final_sd[key] = merged
        else:
            final_sd[key] = base_sd[key].clone()

    del task_vectors, result_tvs, feat_arr, basis_arr, calib_inputs
    gc.collect()
    torch.cuda.empty_cache()

    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 6-h. LOT (Least-squares Optimal Task merging)
#      출처: https://github.com/SunWenJu123/model-merging
# ──────────────────────────────────────────────────────────────────────────────

def run_lot(
    base_sd: dict, expert_sds: List[dict],
    expert_paths: List[str], base_model_path: str,
    scaling_coef: float,
    calib_samples: int, calib_seqlen: int,
    device: str = "cuda:0",
) -> dict:
    """
    LOT (Least-squares Optimal Task merging) — LLM 버전.
    출처: SunWenJu123/model-merging  (lot_merging_utils.py 완전 이식)

    알고리즘:
      원본: model = tv.apply_to(pretrained_checkpoint, scaling_coef=1)  → LOT은 scaling_coef=1로 activation 수집
      각 레이어에 대해 최적 merged task vector를 least-squares로 계산:
        Linear (2D): T_opt = pinv(Σ X_k^T X_k) @ (Σ X_k^T X_k @ T_k^T)  →  T_opt.T
        LayerNorm (1D): weighted average  T_opt = Σ(X_k² * T_k) / Σ(X_k²)

      결과: base + scaling_coef * T_opt
    """
    N = len(expert_sds)

    # task vectors (원본: cur_task_vectors)
    task_vectors = []
    for sd in expert_sds:
        tv = {}
        for key in base_sd:
            if key in sd and base_sd[key].is_floating_point():
                tv[key] = sd[key].float() - base_sd[key].float()
            else:
                tv[key] = None
        task_vectors.append(tv)

    # 원본: opt_tv = copy.deepcopy(sum(task_vectors)); opt_tv /= N  → 평균 초기값
    opt_tv = {}
    for key in base_sd:
        if not base_sd[key].is_floating_point():
            opt_tv[key] = None
            continue
        vals = [tv[key] for tv in task_vectors if tv[key] is not None]
        if vals:
            opt_tv[key] = sum(vals) / len(vals)
        else:
            opt_tv[key] = None

    # C4 calibration 데이터 준비
    print("  [LOT] Preparing C4 calibration data...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    calib_inputs = _prepare_c4_calibration(tokenizer, calib_samples, calib_seqlen, seed=42)
    print(f"    {len(calib_inputs)} sequences × {calib_seqlen} tokens")

    # activation 수집
    # 원본: model = tv.apply_to(pretrained_checkpoint, scaling_coef=1)  → LOT은 1.0
    print("  [LOT] Collecting activations for each expert...")
    feat_arr = []
    for i, path in enumerate(expert_paths):
        print(f"    Expert {i}: {path}")
        feat = _collect_activations_for_expert(
            base_model_path, task_vectors[i],
            calib_inputs, scaling_coef=1.0,   # LOT 원본은 scaling_coef=1
        )
        feat_arr.append(feat)
        gc.collect()
        torch.cuda.empty_cache()

    # 최적 task vector 계산
    print("  [LOT] Computing optimal task vector...")
    compute_device = device if torch.cuda.is_available() else "cpu"

    for key in tqdm(sorted(base_sd.keys()), desc="  LOT optimize"):
        if opt_tv[key] is None:
            continue
        # 원본: if feat_arr[0][name] is None: continue
        if not all(key in feat_arr[t] for t in range(N)):
            continue

        if base_sd[key].ndim == 1 and "norm" in key.lower() and key.endswith(".weight"):
            # 원본 LayerNorm:
            #   numerator += X_k_squared_sum * T_k
            #   denominator += X_k_squared_sum
            #   opt_tv = numerator / denominator
            numerator = torch.zeros_like(base_sd[key].float())
            denominator = torch.zeros_like(base_sd[key].float())
            for t_id in range(N):
                X_k = feat_arr[t_id][key]           # (tokens, dim)
                T_k = task_vectors[t_id][key]        # (dim,)
                X_k_sq_sum = torch.sum(X_k ** 2, dim=0)  # (dim,)
                numerator += X_k_sq_sum * T_k
                denominator += X_k_sq_sum
            # 원본: denominator = torch.where(denominator == 0, torch.tensor(1e-10), denominator)
            denominator = torch.where(denominator == 0, torch.tensor(1e-10), denominator)
            opt_tv[key] = numerator / denominator

        elif base_sd[key].ndim == 2:
            # 원본 Linear:
            #   sum_XTX += X_k.T @ X_k
            #   sum_XTT += X_k.T @ X_k @ T_k.T
            #   T_optimal = torch.linalg.pinv(sum_XTX) @ sum_XTT
            #   opt_tv = T_optimal.T
            f_in = base_sd[key].shape[1]
            f_out = base_sd[key].shape[0]
            sum_XTX = torch.zeros(f_in, f_in, dtype=torch.float32, device=compute_device)
            sum_XTT = torch.zeros(f_in, f_out, dtype=torch.float32, device=compute_device)

            for t_id in range(N):
                X_k = feat_arr[t_id][key].to(compute_device)    # (tokens, f_in)
                T_k = task_vectors[t_id][key].to(compute_device) # (f_out, f_in)
                xtx = X_k.T @ X_k                                # (f_in, f_in)
                sum_XTX += xtx
                sum_XTT += xtx @ T_k.T                           # (f_in, f_out)
                del X_k, T_k, xtx
                torch.cuda.empty_cache()

            T_optimal = torch.linalg.pinv(sum_XTX) @ sum_XTT     # (f_in, f_out)
            opt_tv[key] = T_optimal.T.cpu()                       # (f_out, f_in)
            del sum_XTX, sum_XTT, T_optimal
            torch.cuda.empty_cache()

    # 최종: base + scaling_coef * opt_tv
    # 원본: image_encoder = opt_vector.apply_to(pretrained_checkpoint, scaling_coef=args.scaling_coef_)
    print("  [LOT] Constructing final state dict...")
    final_sd = {}
    for key in base_sd:
        if base_sd[key].is_floating_point() and opt_tv.get(key) is not None:
            final_sd[key] = base_sd[key].float() + scaling_coef * opt_tv[key]
        else:
            final_sd[key] = base_sd[key].clone()

    del task_vectors, feat_arr, opt_tv, calib_inputs
    gc.collect()
    torch.cuda.empty_cache()

    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 6-f. Global Whitened (standalone)
# ──────────────────────────────────────────────────────────────────────────────

def run_global_whitened(
    base_sd: dict, expert_sds: List[dict], k: int,
    device: str = "cpu",
) -> dict:
    """
    Global Whitened merge (standalone):
      base + α_global · Σ τ_wht_i

    energy_direction의 (f) 변형만 단독 실행.
    α_global = mean(per-layer α_wht)
    """
    dev = torch.device(device)
    N = len(expert_sds)

    common = set(base_sd.keys())
    for sd in expert_sds:
        common &= set(sd.keys())
    keys_2d = sorted(key for key in common if base_sd[key].ndim == 2
                      and base_sd[key].dtype in (torch.float32, torch.float16, torch.bfloat16))
    keys_other = [key for key in base_sd if key not in set(keys_2d)]

    # Pass 1: compute per-layer tau_wht (unscaled) and alpha_wht
    tau_wht_dict = {}
    alpha_wht_list = []

    with torch.no_grad():
        for key in tqdm(keys_2d, desc="Global Whitened (pass 1)"):
            W_base = base_sd[key].float().to(dev)
            min_dim = min(W_base.shape)
            task_vecs = [sd[key].float().to(dev) - W_base for sd in expert_sds]
            D_sq = sum(t.norm().item() ** 2 for t in task_vecs)

            k_eff = min(k, min_dim // N)
            if k_eff == 0:
                tau_wht_dict[key] = (sum(task_vecs) / N).cpu()
                alpha_wht_list.append(1.0)
                continue

            Us, Ss, Vs = [], [], []
            for tau in task_vecs:
                U, S, Vh = torch.linalg.svd(tau, full_matrices=False)
                Us.append(U[:, :k_eff])
                Ss.append(S[:k_eff])
                Vs.append(Vh[:k_eff, :].T)

            U_hat = polar_factor(torch.cat(Us, dim=1))
            V_hat = polar_factor(torch.cat(Vs, dim=1))

            tau_wht = torch.zeros_like(W_base)
            for i in range(N):
                s, e = i * k_eff, (i + 1) * k_eff
                tau_wht += U_hat[:, s:e] @ torch.diag(Ss[i]) @ V_hat[:, s:e].T

            wht_sq = tau_wht.norm().item() ** 2
            alpha_wht = math.sqrt(D_sq / (wht_sq + 1e-12)) if wht_sq > 1e-12 else 1.0
            alpha_wht_list.append(alpha_wht)
            tau_wht_dict[key] = tau_wht.cpu()

    # Pass 2: apply global alpha
    alpha_global = sum(alpha_wht_list) / len(alpha_wht_list)
    print(f"  k={k}: global α = {alpha_global:.6f}")

    final_sd = {}
    with torch.no_grad():
        for key in keys_2d:
            W_base = base_sd[key].float()
            final_sd[key] = W_base + alpha_global * tau_wht_dict[key]

    for key in keys_other:
        if base_sd[key].is_floating_point():
            W_base = base_sd[key].float()
            tvs = [sd[key].float() - W_base for sd in expert_sds if key in sd
                   if sd[key].is_floating_point()]
            final_sd[key] = (W_base + sum(tvs) / len(tvs)) if tvs else W_base.clone()
        else:
            final_sd[key] = base_sd[key].clone()

    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 9. Iso-C (Isotropic Merging in Common Subspace)
# ──────────────────────────────────────────────────────────────────────────────

def run_iso_c(
    base_sd: dict, expert_sds: List[dict],
    alpha: float, device: str,
) -> dict:
    """
    Iso-C: Isotropic Merging in Common Subspace.
    출처: "No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces"
           https://github.com/danielm1405/iso-merging (ICML 2025)

    각 2D 레이어에 대해:
      1. Task vector 합산: W_sum = Σ_t τ_t  (Task Arithmetic sum)
      2. SVD: W_sum = U S V^T
      3. Singular value spectrum 평탄화: S_iso = mean(S) · ones_like(S)
      4. 재구성: W_iso = U · diag(S_iso) · V^T
      5. 스케일 적용: base + alpha · W_iso

    2D가 아닌 레이어(bias, layernorm 등): task vector 단순 평균.
    """
    keys = get_numeric_keys(base_sd, expert_sds)
    N = len(expert_sds)
    final_sd = {}

    with torch.no_grad():
        for key in tqdm(keys, desc="Iso-C"):
            tvs = [sd[key].float().to(device) - base_sd[key].float().to(device)
                   for sd in expert_sds]

            # 1. average task vectors (원본: sum(tvs) / len(tvs))
            merged_tv = sum(tvs) / N

            if base_sd[key].dim() == 2:
                # 2. un-average → sum (원본: new_vector[key] *= len(tvs))
                merged_tv = merged_tv * N

                # 3. SVD
                U, S, V = torch.linalg.svd(merged_tv, full_matrices=False)

                # 4. singular value spectrum 평탄화 (isotropic)
                S_iso = torch.ones_like(S) * S.mean()

                # 5. 재구성
                merged_tv = torch.linalg.multi_dot((U, torch.diag(S_iso), V))

            final_sd[key] = (base_sd[key].float().to(device) + alpha * merged_tv).cpu()

    for key in base_sd:
        if key not in final_sd:
            final_sd[key] = base_sd[key]

    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 10. Iso-CTS (Isotropic Merging in Common and Task-Specific Subspaces)
# ──────────────────────────────────────────────────────────────────────────────

def run_iso_cts(
    base_sd: dict, expert_sds: List[dict],
    alpha: float, common_space_fraction: float, device: str,
) -> dict:
    """
    Iso-CTS: Isotropic Merging in Common and Task-Specific Subspaces.
    출처: "No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces"
           https://github.com/danielm1405/iso-merging (ICML 2025)

    각 2D 레이어에 대해:
      1. Common subspace: Σ_t τ_t 의 SVD 상위 k_c 개 singular direction
      2. Task-specific subspace: 각 τ_t 에서 common 방향을 제거한 잔차의
         상위 k_ts 개 singular direction  (T 개 task × k_ts dims)
      3. task-specific 블록(앞) + common 블록(뒤) 연결
      4. SVD polar decomposition 으로 U, V 재직교화
      5. Singular value spectrum 평탄화 (mean → isotropic)
      6. 재구성 후 alpha 스케일 적용

    2D가 아닌 레이어: task vector running average.

    Args:
        common_space_fraction: common subspace 에 할당할 singular dimension 비율
                               (기본값: 0.8, 논문 권장)
    """
    keys = get_numeric_keys(base_sd, expert_sds)
    N = len(expert_sds)
    final_sd = {}

    with torch.no_grad():
        for key in tqdm(keys, desc="Iso-CTS"):
            shape_ = base_sd[key].shape
            is_2d_matrix = (base_sd[key].dim() == 2)

            # ── Non-2D: running average of task vectors ──
            if not is_2d_matrix:
                result = None
                for i, sd in enumerate(expert_sds):
                    tv = sd[key].float().to(device) - base_sd[key].float().to(device)
                    if i == 0:
                        result = tv.clone()
                    else:
                        result = result + (tv - result) / (i + 1)
                final_sd[key] = (base_sd[key].float().to(device) + alpha * result).cpu()
                continue

            # ── 2D: Iso-CTS procedure ──

            # Step 1: task vector 합산 (common subspace 는 TA sum 기반)
            tvs = [sd[key].float().to(device) - base_sd[key].float().to(device)
                   for sd in expert_sds]
            combined_w = sum(tvs)

            min_dim = min(shape_)

            # Step 2: common / task-specific split 크기 결정
            #   - task-specific total 이 N 의 배수가 되도록 조정
            common_space_index_s = int(min_dim * common_space_fraction)
            _task_specific_total = round(
                (min_dim - common_space_index_s) / N
            ) * N
            common_space_index_s = min_dim - _task_specific_total

            n_dims_per_task = int((min_dim - common_space_index_s) / N)

            # Step 3: combined_w 의 SVD → common subspace (상위 singular directions)
            u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
            common_space_u = u[:, :common_space_index_s]   # [M, k_c]
            common_space_s = s[:common_space_index_s]       # [k_c]
            common_space_v = v[:common_space_index_s, :]    # [k_c, D]

            # Step 4: 각 task 의 task-specific subspace 추출
            #   common 방향을 orthogonal projection 으로 제거 후 잔차 SVD
            combined_space_u = None
            combined_space_s = None
            combined_space_v = None

            for i, tv in enumerate(tvs):
                # common subspace 성분 제거 (직교 투영)
                w_ts = tv - common_space_u @ (common_space_u.T @ tv)

                # 잔차(task-specific) 행렬의 SVD
                u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)

                if i == 0:
                    combined_space_u = torch.zeros_like(u_ts, device=device)
                    combined_space_s = torch.zeros_like(s_ts, device=device)
                    combined_space_v = torch.zeros_like(v_ts, device=device)

                # 이 task 의 상위 n_dims_per_task 개 singular component 를 해당 슬롯에 배치
                combined_space_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = u_ts[:, :n_dims_per_task]
                combined_space_s[i * n_dims_per_task : (i + 1) * n_dims_per_task]    = s_ts[:n_dims_per_task]
                combined_space_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = v_ts[:n_dims_per_task, :]

            # Step 5: common subspace 블록을 끝에 추가
            ts_end = N * n_dims_per_task
            combined_space_u[:, ts_end : ts_end + common_space_index_s] = common_space_u
            combined_space_s[ts_end : ts_end + common_space_index_s]    = common_space_s
            combined_space_v[ts_end : ts_end + common_space_index_s, :] = common_space_v

            # Step 6: U, V 재직교화 — SVD polar decomposition (nearest orthogonal matrix)
            #   task-specific + common 벡터들의 연결로 직교성이 깨질 수 있음
            #   ill-conditioned 행렬 대비 소수 노이즈 fallback
            try:
                u_uu, _, v_uu = torch.linalg.svd(combined_space_u, full_matrices=False)
            except torch._C._LinAlgError:
                combined_space_u = combined_space_u + 1e-6 * torch.randn_like(combined_space_u)
                u_uu, _, v_uu = torch.linalg.svd(combined_space_u, full_matrices=False)
            try:
                u_vv, _, v_vv = torch.linalg.svd(combined_space_v, full_matrices=False)
            except torch._C._LinAlgError:
                combined_space_v = combined_space_v + 1e-6 * torch.randn_like(combined_space_v)
                u_vv, _, v_vv = torch.linalg.svd(combined_space_v, full_matrices=False)
            combined_space_u = u_uu @ v_uu
            combined_space_v = u_vv @ v_vv

            # Step 7: singular value spectrum 평탄화 (isotropic)
            combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()

            # Step 8: merged task vector 재구성
            merged_tv = torch.linalg.multi_dot((
                combined_space_u,
                torch.diag(combined_space_s),
                combined_space_v,
            ))

            final_sd[key] = (base_sd[key].float().to(device) + alpha * merged_tv).cpu()

    for key in base_sd:
        if key not in final_sd:
            final_sd[key] = base_sd[key]

    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 10-b. SVD Truncation (task vector rank-k truncation + simple sum)
# ──────────────────────────────────────────────────────────────────────────────

def _svd_truncate_tensor(delta: torch.Tensor, k: int) -> torch.Tensor:
    """2D delta를 SVD 후 상위 k개 singular value만 남겨 재구성. 1D는 그대로 반환."""
    if delta.dim() < 2:
        return delta.clone()
    u, s, vt = torch.linalg.svd(delta, full_matrices=False)
    rank = min(k, s.shape[0])
    return u[:, :rank] @ torch.diag(s[:rank]) @ vt[:rank, :]


def run_svd_truncation(
    base_sd: dict, expert_sds: List[dict], k: int,
) -> dict:
    """
    SVD Truncation Merge.

    각 expert의 task vector (expert - base)를 SVD 후 상위 k개 singular value만
    남기고(truncation), 그 결과를 단순 합산하여 base에 더함.

    수식: merged = base + Σ trunc_k(expert_i - base)

    Args:
        base_sd    : 베이스 모델 state dict
        expert_sds : 전문가 모델 state dict 리스트
        k          : 유지할 singular value 개수 (rank)
    """
    keys = get_numeric_keys(base_sd, expert_sds)
    N = len(expert_sds)

    final_sd = {}
    with torch.no_grad():
        for key in tqdm(keys, desc=f"  SVD Truncation (k={k})"):
            W_base = base_sd[key].float()
            tau_trunc_sum = torch.zeros_like(W_base)
            for i in range(N):
                delta = expert_sds[i][key].float() - W_base
                tau_trunc_sum.add_(_svd_truncate_tensor(delta, k))
            final_sd[key] = W_base + tau_trunc_sum

    for key in base_sd:
        if key not in final_sd:
            final_sd[key] = base_sd[key]

    return final_sd


# ──────────────────────────────────────────────────────────────────────────────
# 11. RAM (Reinforced Agentic Merge)
#     원본: https://github.com/xiangchi-yuan/mrl  (ram-main.py)
# ──────────────────────────────────────────────────────────────────────────────

def run_ram(
    base_sd: dict, expert_sds: List[dict],
    threshold: float = 1e-5,
    device: str = "cpu",
) -> dict:
    """
    RAM (Reinforced Agentic Merge) — 기본 overlap-aware averaging.

    원본 함수: agentic_reinforcement_merge (ram-main.py)

    알고리즘:
      1. task vector 계산: Δ_i = expert_i - base
      2. change mask: |Δ_i| > threshold
      3. 변경된 파라미터만 평균: avg = Σ(Δ_i * mask_i) / Σ mask_i
      4. merged = base + avg

    GPU 사용 시 per-key로 GPU 이동하여 VRAM 절약.
    """
    keys = get_numeric_keys(base_sd, expert_sds)
    n = len(expert_sds)

    # ── task vector 계산 (CPU에 유지) ──
    print(f"  Computing {n} task vectors...")
    task_vecs = []
    for sd in expert_sds:
        tv = {}
        for k in keys:
            tv[k] = sd[k].float() - base_sd[k].float()
        task_vecs.append(tv)

    # ── 머징 (per-key GPU 이동) ──
    merged_sd = {}
    with torch.no_grad():
        for k in tqdm(keys, desc="RAM"):
            diffs = torch.stack([tv[k] for tv in task_vecs], dim=0).to(device)  # (n, *shape)
            change_mask = (diffs.abs() > threshold)
            change_mask_f = change_mask.float()

            sum_diff = (diffs * change_mask_f).sum(dim=0)
            count = change_mask_f.sum(dim=0)

            denom = torch.clamp(count, min=1.0)
            avg_diff = sum_diff / denom
            diff_final = torch.where(count > 0, avg_diff, torch.zeros_like(sum_diff))

            merged_sd[k] = (base_sd[k].float().to(device) + diff_final).cpu()

    for k in base_sd:
        if k not in merged_sd:
            merged_sd[k] = base_sd[k]

    return merged_sd


def run_ram_plus(
    base_sd: dict, expert_sds: List[dict],
    threshold: float = 1e-5,
    rescale_factor: float = 1.05,
    device: str = "cpu",
) -> dict:
    """
    RAM+ (ARM-R-V2) — overlap 비율 기반 rescaling 적용.

    원본 함수: agentic_reinforcement_merge_rescale_v2 (ram-main.py)

    알고리즘:
      Phase 1 — rescale factor 계산:
        - 각 task j에 대해:
          changed_j = task j가 변경한 파라미터 수
          overlap_j = 그 중 2개 이상 task가 동시에 변경한 수
          ratio_j   = overlap_j / (changed_j - overlap_j)
          r_j       = 1.0 + (r - 1.0) * min(2, min(1.0, ratio_j))

      Phase 2 — 머징:
        - overlap 영역 (count >= 2): 평균
        - non-overlap 영역 (count == 1): rescale factor 적용한 weighted sum
        - merged = base + diff_final

    GPU 사용 시 per-key로 GPU 이동하여 VRAM 절약.
    """
    keys = get_numeric_keys(base_sd, expert_sds)
    n = len(expert_sds)

    r = float(rescale_factor)
    if r <= 1.0:
        r = 1.0

    # ── task vector 계산 (CPU에 유지) ──
    print(f"  Computing {n} task vectors...")
    task_vecs = []
    for sd in expert_sds:
        tv = {}
        for k in keys:
            tv[k] = sd[k].float() - base_sd[k].float()
        task_vecs.append(tv)

    # ── Phase 1: overlap 통계 및 rescale factor 계산 (per-key GPU 이동) ──
    print("  Computing overlap statistics...")
    changed_counts = [0] * n
    overlap_counts = [0] * n

    for name in tqdm(keys, desc="RAM+ overlap stats"):
        diffs = torch.stack([tv[name] for tv in task_vecs], dim=0).to(device)
        change_mask = (diffs.abs() > threshold)
        sum_change = change_mask.sum(dim=0)

        overlap_any = (sum_change >= 2)

        change_flat = change_mask.view(n, -1)
        overlap_flat = overlap_any.view(-1)

        for j in range(n):
            cj = change_flat[j]
            changed_counts[j] += cj.sum().item()
            overlap_counts[j] += (cj & overlap_flat).sum().item()

    overlap_ratios = []
    rescales = []
    for j in range(n):
        if changed_counts[j] == 0:
            ratio = 0.0
        else:
            ratio = overlap_counts[j] / (changed_counts[j] - overlap_counts[j])
        overlap_ratios.append(ratio)

        rescale_j = 1.0 + (r - 1.0) * min(2, min(1.0, float(ratio)))
        rescales.append(rescale_j)

    print(f"  [OverlapAware] overlap_ratios per task: {overlap_ratios}")
    print(f"  [OverlapAware] rescale per task: {rescales}")

    # ── Phase 2: rescale 적용 머징 (per-key GPU 이동) ──
    merged_sd = {}
    with torch.no_grad():
        for name in tqdm(keys, desc="RAM+ merge"):
            diffs = torch.stack([tv[name] for tv in task_vecs], dim=0).to(device)

            change_mask = (diffs.abs() > threshold)
            change_mask_f = change_mask.float()

            sum_diff = (diffs * change_mask_f).sum(dim=0)
            count = change_mask_f.sum(dim=0)

            denom = torch.clamp(count, min=1.0)
            avg_diff = sum_diff / denom

            zero = torch.zeros_like(sum_diff)
            non_overlap_mask = (count == 1)
            overlap_mask = (count >= 2)

            rescales_tensor = torch.tensor(
                rescales,
                dtype=diffs.dtype,
                device=diffs.device,
            ).view((n,) + (1,) * (diffs.dim() - 1))

            weighted_sum = (diffs * change_mask_f * rescales_tensor).sum(dim=0)

            diff_final = zero
            diff_final = torch.where(overlap_mask, avg_diff, diff_final)
            diff_final = torch.where(non_overlap_mask, weighted_sum, diff_final)

            merged_sd[name] = (base_sd[name].float().to(device) + diff_final).cpu()

    for k in base_sd:
        if k not in merged_sd:
            merged_sd[k] = base_sd[k]

    return merged_sd


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified Model Merging Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 공통 인자
    p.add_argument("--method", required=True,
        choices=["task_arithmetic", "ties", "dare", "star", "cart", "tsv", "fisher",
                 "whitened", "whitened_noscale", "per_tv_renorm", "capped_pertv",
                 "energy_direction", "global_whitened", "cat", "lot",
                 "iso_c", "iso_cts", "ram", "ram_plus", "svd_truncation",
                 "rmt_whitened", "rmt_per_tv", "rmt_optimal"],
        help="머징 기법 선택")
    p.add_argument("--base_model", required=True,
        help="베이스(프리트레인) 모델 경로 또는 HF 이름")
    p.add_argument("--expert_models", nargs="+", required=True,
        help="파인튠된 전문가 모델 경로들 (2개 이상)")
    p.add_argument("--save_dir", required=True,
        help="머지 결과물 저장 디렉토리")
    p.add_argument("--cache_dir", default=None,
        help="HuggingFace 캐시 디렉토리 (기본: 환경변수 또는 자동감지)")

    # Task Arithmetic / Fisher 공통
    p.add_argument("--lambdas", type=float, nargs="+", default=None,
        help="[task_arithmetic/fisher] 전문가별 스케일 계수. "
             "task_arithmetic 기본: 1/N (uniform mean). fisher 기본: 모두 1.0.")

    # TIES
    p.add_argument("--lamda", type=float, default=1.0,
        help="[ties] 머지된 task vector 스케일 (기본: 1.0)")
    p.add_argument("--density", type=float, default=0.2,
        help="[ties] TRIM 단계 유지 비율 top-K%% (기본: 0.2)")
    p.add_argument("--sign_method", default="mass",
        choices=["mass", "normfrac", "normmass"],
        help="[ties] 부호 결정 방식 (기본: mass)")
    p.add_argument("--merge_func", default="dis-mean",
        choices=["dis-mean", "dis-sum", "dis-max", "mean", "sum"],
        help="[ties] 집계 방식 (기본: dis-mean)")

    # DARE
    p.add_argument("--weight_mask_rate", type=float, default=0.9,
        help="[dare] delta 드롭 비율 (기본: 0.9, 논문 권장값)")
    p.add_argument("--use_rescale", action="store_true", default=True,
        help="[dare] 생존 원소를 1/(1-mask_rate)로 재스케일 (기본: True)")
    p.add_argument("--no_rescale", dest="use_rescale", action="store_false",
        help="[dare] 재스케일 비활성화")
    p.add_argument("--mask_strategy", default="random",
        choices=["random", "magnitude"],
        help="[dare] 드롭 전략: random(Bernoulli) | magnitude(소값 제거) (기본: random)")
    p.add_argument("--dare_merge_method", default="task_arithmetic",
        choices=["task_arithmetic", "ties"],
        help="[dare] DARE 희소화 후 적용할 머징 방법 (기본: task_arithmetic)")

    # STAR
    p.add_argument("--eta", type=float, default=40.0,
        help="[star] nuclear norm 유지 비율 %% (기본: 40.0, 낮을수록 더 압축)")

    # CART
    p.add_argument("--prior", type=float, default=2.0,
        help="[cart] 로우랭크 합의 스케일 계수 (기본: 2.0, 논문: 8-task=2.0 / 14-task=1.5 / 20-task=1.9)")
    p.add_argument("--rank_ratio", type=float, default=0.12,
        help="[cart] SVD truncation rank 비율 (기본: 0.12, 논문: 8-task=0.12 / 14-task=0.16 / 20-task=0.32)")

    # TSV / Iso-C / Iso-CTS
    p.add_argument("--alpha", type=float, default=1.0,
        help="[tsv/iso_c/iso_cts] 머지된 task vector 스케일 (기본: 1.0)")
    p.add_argument("--k", type=int, default=None,
        help="[tsv] 전문가당 유지할 singular value 수 (고정)")
    p.add_argument("--sv_reduction", type=float, default=None,
        help="[tsv] 유지할 singular value 비율 (기본: 1/N)")

    # Whitened / TSV 공통 (k_list)
    p.add_argument("--k_list", type=int, nargs="+", default=[256],
        help="[whitened/per_tv_renorm/capped_pertv/energy_direction/svd_truncation] SVD truncation rank 목록 (기본: 256)")

    # RMT
    p.add_argument("--k_max_cap", type=int, default=1024,
        help="[rmt_*] RMT auto-k의 상한 (메모리 제한, 기본: 1024)")
    p.add_argument("--k_min_floor", type=int, default=0,
        help="[rmt_*] RMT auto-k의 하한 (기본: 0, 예: 128)")

    # Iso-CTS
    p.add_argument("--common_space_fraction", type=float, default=0.8,
        help="[iso_cts] common subspace 에 할당할 singular dimension 비율 (기본: 0.8, 논문 권장값)")

    # Fisher
    p.add_argument("--epsilon", type=float, default=1e-12,
        help="[fisher] 분모 안정화 상수 (기본: 1e-12)")
    p.add_argument("--fisher_files", type=str, nargs="+", default=None,
        help="[fisher] 미리 계산된 Fisher .pt 파일들")
    p.add_argument("--calib_samples", type=int, default=64,
        help="[fisher] Fisher 추정 calibration 샘플 수 (기본: 64)")
    p.add_argument("--calib_seqlen", type=int, default=128,
        help="[fisher] calibration 시퀀스 길이 (기본: 128)")
    p.add_argument("--fisher_device", type=str, default="cuda:0",
        help="[fisher] Fisher 계산 디바이스 (기본: cuda:0)")
    p.add_argument("--save_fisher", action="store_true",
        help="[fisher] 계산된 Fisher 파일 저장")

    # CAT / LOT 공통
    p.add_argument("--scaling_coef", type=float, default=0.4,
        help="[cat/lot] merged task vector 스케일 계수 (기본: 0.4, CAT 논문 기본값)")
    p.add_argument("--cat_calib_samples", type=int, default=2,
        help="[cat/lot] activation 수집용 calibration 샘플 수 (기본: 2, 논문 exp_size)")
    p.add_argument("--cat_calib_seqlen", type=int, default=128,
        help="[cat/lot] calibration 시퀀스 길이 (기본: 128)")

    # CAT 전용
    p.add_argument("--cat_ratio", type=float, default=4,
        help="[cat] conflict basis 선택 기준: -1=전체 양수, 0<r<1=에너지 비율, >=1=고정 rank (기본: 4)")
    p.add_argument("--cat_lambd", type=float, default=5.0,
        help="[cat] 자기 task 대비 타 task conflict 가중치 (기본: 5.0)")
    p.add_argument("--cat_ratio_ln", type=int, default=4,
        help="[cat] LayerNorm conflict masking top-k 차원 수 (0=비활성화, 기본: 4)")
    p.add_argument("--cat_lambd_ln", type=float, default=10.0,
        help="[cat] LayerNorm conflict 가중치 (기본: 10.0)")
    p.add_argument("--cat_ratio_bias", type=int, default=4,
        help="[cat] bias/1D conflict masking top-k 차원 수 (0=비활성화, 기본: 4)")

    # RAM / RAM+
    p.add_argument("--ram_threshold", type=float, default=1e-5,
        help="[ram/ram_plus] 파라미터 변경 감지 threshold (기본: 1e-5)")
    p.add_argument("--rescale_factor", type=float, default=1.05,
        help="[ram_plus] overlap 기반 rescale factor r (기본: 1.05, 원본 CLI default)")

    # 계산 디바이스 (TIES, TSV)
    p.add_argument("--device", type=str, default="cpu",
        help="[ties/tsv] 계산 디바이스 (기본: cpu)")

    return p


def _validate_save_dir(save_dir: str) -> str:
    """save_dir이 REQUIRED_SAVE_ROOT 하위인지 확인. KT_SAVE_ROOT='' 이면 검증 skip."""
    if not REQUIRED_SAVE_ROOT:
        return save_dir
    resolved = str(Path(save_dir).resolve())
    required = str(Path(REQUIRED_SAVE_ROOT).resolve())
    if not (resolved == required or resolved.startswith(required + "/")):
        raise ValueError(
            f"\n[오류] --save_dir must be under '{REQUIRED_SAVE_ROOT}'.\n"
            f"  got: {save_dir}\n"
            f"  to disable check: export KT_SAVE_ROOT=\"\""
        )
    return save_dir


def main():
    parser = build_parser()
    args = parser.parse_args()

    N = len(args.expert_models)
    if N < 2:
        parser.error("전문가 모델은 최소 2개 이상 필요합니다.")

    # ── save_dir 경로 강제 검증 ──
    try:
        _validate_save_dir(args.save_dir)
    except ValueError as e:
        parser.error(str(e))

    setup_cache(args.cache_dir)

    print("=" * 60)
    print(f"  Method: {args.method.upper()}")
    print("=" * 60)
    print(f"  Base:     {args.base_model}")
    for i, m in enumerate(args.expert_models):
        print(f"  Expert {i}: {m}")
    print(f"  Save:     {args.save_dir}")
    print("=" * 60)

    # ── 모델 로드 ──
    print("\n[1/3] Loading models...")
    base_sd = load_state_dict(args.base_model)
    expert_sds = [load_state_dict(m) for m in args.expert_models]

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 머징 ──
    print(f"\n[2/3] Merging ({args.method})...")

    if args.method == "task_arithmetic":
        # Uniform mean: λ_i = 1/N (override via --lambdas).
        lambdas = args.lambdas or [1.0 / N] * N
        if len(lambdas) != N:
            parser.error(f"--lambdas 개수({len(lambdas)})가 모델 수({N})와 다릅니다.")
        print(f"  λ = {lambdas}")
        final_sd = run_task_arithmetic(base_sd, expert_sds, lambdas)

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "ties":
        print(f"  λ={args.lamda}, density={args.density}, sign={args.sign_method}, func={args.merge_func}")
        final_sd = run_ties(base_sd, expert_sds, args.lamda, args.density,
                            args.sign_method, args.merge_func, args.device)

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "dare":
        lambdas = args.lambdas or [1.0] * N
        if len(lambdas) != N:
            parser.error(f"--lambdas 개수({len(lambdas)})가 모델 수({N})와 다릅니다.")
        print(f"  mask_rate={args.weight_mask_rate}, rescale={args.use_rescale}, "
              f"strategy={args.mask_strategy}, merge_method={args.dare_merge_method}")
        if args.dare_merge_method == "task_arithmetic":
            print(f"  λ = {lambdas}")
        else:
            print(f"  ties: λ={args.lamda}, density={args.density}, "
                  f"sign={args.sign_method}, func={args.merge_func}")
        final_sd = run_dare(
            base_sd, expert_sds, lambdas,
            weight_mask_rate=args.weight_mask_rate,
            use_rescale=args.use_rescale,
            mask_strategy=args.mask_strategy,
            merge_method=args.dare_merge_method,
            ties_lamda=args.lamda,
            ties_density=args.density,
            ties_sign_method=args.sign_method,
            ties_merge_func=args.merge_func,
            device=args.device,
        )

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "star":
        # 논문 기본: 균등 평균 (1/N). --lambdas로 커스텀 가중치 지정 가능
        lambdas = args.lambdas or [1.0 / N] * N
        if len(lambdas) != N:
            parser.error(f"--lambdas 개수({len(lambdas)})가 모델 수({N})와 다릅니다.")
        print(f"  eta={args.eta}, λ={lambdas}")
        final_sd = run_star(base_sd, expert_sds, args.eta, lambdas)

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "cart":
        print(f"  prior={args.prior}, rank_ratio={args.rank_ratio}")
        final_sd = run_cart(base_sd, expert_sds, args.prior, args.rank_ratio)

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "tsv":
        if args.k is not None and args.sv_reduction is not None:
            parser.error("--k와 --sv_reduction은 동시에 사용할 수 없습니다.")
        print(f"  α={args.alpha}, k={args.k}, sv_reduction={args.sv_reduction}")
        final_sd = run_tsv(base_sd, expert_sds, args.alpha, args.k, args.sv_reduction, args.device)

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "fisher":
        lambdas = args.lambdas or [1.0] * N
        if len(lambdas) != N:
            parser.error(f"--lambdas 개수({len(lambdas)})가 모델 수({N})와 다릅니다.")
        final_sd = run_fisher(
            base_sd, expert_sds, args.expert_models,
            lambdas, args.epsilon,
            args.fisher_files, args.calib_samples, args.calib_seqlen,
            args.fisher_device, args.save_fisher, args.save_dir,
        )

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method in ("whitened", "whitened_noscale", "per_tv_renorm", "capped_pertv"):
        mode_map = {
            "whitened": "scaled",
            "whitened_noscale": "noscale",
            "per_tv_renorm": "per_tv",
            "capped_pertv": "capped",
        }
        mode = mode_map[args.method]
        k_list = sorted(args.k_list)
        print(f"  mode={mode}, k_list={k_list}")

        merged_per_k, stats_per_k = _whitened_merge_core(base_sd, expert_sds, k_list, mode)

        del expert_sds
        gc.collect()

        # 통계 저장
        for k_val in k_list:
            sp = os.path.join(args.save_dir, f"stats_{args.method}_k{k_val}.json")
            with open(sp, "w") as f:
                json.dump(stats_per_k[k_val], f, indent=2, default=str)
            print(f"  Stats → {sp}")

        print("\n[3/3] Saving...")
        for k_val in k_list:
            out = os.path.join(args.save_dir, f"{args.method}_k{k_val}")
            save_model(args.base_model, merged_per_k[k_val], out)
            merged_per_k[k_val].clear()
            gc.collect()

    elif args.method == "energy_direction":
        k_list = sorted(args.k_list)
        print(f"  k_list={k_list}  (변형 6종 × {len(k_list)} k값 생성)")

        sd_direct, sd_scaled, sd_trunc, sd_wht, sd_noscale, sd_gwht, stats_per_k = \
            run_energy_direction(base_sd, expert_sds, k_list)

        del expert_sds
        gc.collect()

        # 통계 저장
        for k_val in k_list:
            sp = os.path.join(args.save_dir, f"stats_energy_direction_k{k_val}.json")
            with open(sp, "w") as f:
                json.dump(stats_per_k[k_val], f, indent=2, default=str)

        print("\n[3/3] Saving all variants...")
        save_model(args.base_model, sd_direct, os.path.join(args.save_dir, "direct_sum"))
        sd_direct.clear(); gc.collect()

        save_model(args.base_model, sd_scaled, os.path.join(args.save_dir, "scaled_sum"))
        sd_scaled.clear(); gc.collect()

        for k_val in k_list:
            save_model(args.base_model, sd_trunc[k_val],   os.path.join(args.save_dir, f"truncated_sum_k{k_val}"))
            save_model(args.base_model, sd_wht[k_val],     os.path.join(args.save_dir, f"whitened_k{k_val}"))
            save_model(args.base_model, sd_noscale[k_val], os.path.join(args.save_dir, f"whitened_noscale_k{k_val}"))
            save_model(args.base_model, sd_gwht[k_val],    os.path.join(args.save_dir, f"global_whitened_k{k_val}"))
            sd_trunc[k_val].clear(); sd_wht[k_val].clear()
            sd_noscale[k_val].clear(); sd_gwht[k_val].clear()
            gc.collect()

        print(f"\n  생성된 변형:")
        print(f"    direct_sum/       — (a) base + Σ τ_i")
        print(f"    scaled_sum/       — (b) base + α·Σ τ_i (per-layer α)")
        for k_val in k_list:
            print(f"    truncated_sum_k{k_val}/  — (c) base + Σ τ_i_trunc")
            print(f"    whitened_k{k_val}/        — (d) base + α_wht·Σ τ_wht_i")
            print(f"    whitened_noscale_k{k_val}/ — (e) base + Σ τ_wht_i")
            print(f"    global_whitened_k{k_val}/  — (f) base + α_global·Σ τ_wht_i")

    elif args.method == "global_whitened":
        k_val = args.k_list[0]
        print(f"  k={k_val}")
        final_sd = run_global_whitened(base_sd, expert_sds, k_val, device=args.device)

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "cat":
        print(f"  scaling_coef={args.scaling_coef}, ratio={args.cat_ratio}, λ={args.cat_lambd}")
        print(f"  ratio_ln={args.cat_ratio_ln}, λ_ln={args.cat_lambd_ln}, ratio_bias={args.cat_ratio_bias}")
        print(f"  calib: samples={args.cat_calib_samples}, seqlen={args.cat_calib_seqlen}")
        final_sd = run_cat(
            base_sd, expert_sds,
            expert_paths=args.expert_models, base_model_path=args.base_model,
            scaling_coef=args.scaling_coef,
            calib_samples=args.cat_calib_samples, calib_seqlen=args.cat_calib_seqlen,
            cat_ratio=args.cat_ratio, cat_lambd=args.cat_lambd,
            cat_ratio_ln=args.cat_ratio_ln, cat_lambd_ln=args.cat_lambd_ln,
            cat_ratio_bias=args.cat_ratio_bias,
            device=args.fisher_device,
        )

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "lot":
        print(f"  scaling_coef={args.scaling_coef}")
        print(f"  calib: samples={args.cat_calib_samples}, seqlen={args.cat_calib_seqlen}")
        final_sd = run_lot(
            base_sd, expert_sds,
            expert_paths=args.expert_models, base_model_path=args.base_model,
            scaling_coef=args.scaling_coef,
            calib_samples=args.cat_calib_samples, calib_seqlen=args.cat_calib_seqlen,
            device=args.fisher_device,
        )

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "iso_c":
        print(f"  alpha={args.alpha}")
        final_sd = run_iso_c(base_sd, expert_sds, args.alpha, args.device)

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "iso_cts":
        print(f"  alpha={args.alpha}, common_space_fraction={args.common_space_fraction}")
        final_sd = run_iso_cts(
            base_sd, expert_sds, args.alpha, args.common_space_fraction, args.device,
        )

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "svd_truncation":
        k_val = args.k_list[0]
        print(f"  k={k_val}")
        final_sd = run_svd_truncation(base_sd, expert_sds, k_val)

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "ram":
        print(f"  threshold={args.ram_threshold}, device={args.device}")
        final_sd = run_ram(base_sd, expert_sds, threshold=args.ram_threshold, device=args.device)

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method == "ram_plus":
        print(f"  threshold={args.ram_threshold}, rescale_factor={args.rescale_factor}, device={args.device}")
        final_sd = run_ram_plus(
            base_sd, expert_sds,
            threshold=args.ram_threshold,
            rescale_factor=args.rescale_factor,
            device=args.device,
        )

        del expert_sds
        gc.collect()
        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    elif args.method in ("rmt_whitened", "rmt_per_tv", "rmt_optimal"):
        rmt_mode_map = {
            "rmt_whitened": ("scaled", "energy"),
            "rmt_per_tv": ("per_tv", "energy"),
            "rmt_optimal": ("per_tv", "optimal"),
        }
        mode, shrinkage = rmt_mode_map[args.method]
        print(f"  RMT mode={mode}, shrinkage={shrinkage}, k_max_cap={args.k_max_cap}")

        final_sd, rmt_stats = run_rmt_merge(
            base_sd, expert_sds,
            mode=mode, shrinkage=shrinkage,
            k_max_cap=args.k_max_cap, k_min_floor=args.k_min_floor,
        )

        del expert_sds
        gc.collect()

        # 통계 저장
        sp = os.path.join(args.save_dir, f"stats_{args.method}.json")
        with open(sp, "w") as f:
            json.dump(rmt_stats, f, indent=2, default=str)
        print(f"  Stats → {sp}")

        print("\n[3/3] Saving...")
        save_model(args.base_model, final_sd, args.save_dir)

    del base_sd
    gc.collect()

    print(f"\n완료! 결과물: {args.save_dir}/")


if __name__ == "__main__":
    main()
