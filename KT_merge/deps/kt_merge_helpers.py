"""Minimal helpers used by KT_merge scripts.

Extracted from a larger experiment file (`exp2_subspace_decomp.py`,
~945 lines, mostly eval-runner / plotting / multi-experiment orchestration
unrelated to merging) — only the 5 functions actually imported by the
KT_merge pipeline are kept here.

Public API:
  _resolve_model_path  — resolve HF id or local path; returns local dir
  load_state_dict      — read safetensors shards (or HF) into one dict
  save_model           — write merged state dict + base config/tokenizer
  get_2d_key_set       — keys of 2D float tensors (the matrices to merge)
  compute_task_vector  — τ = expert - base, float32, numeric-key only
"""
from __future__ import annotations
import gc
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import torch


def _resolve_model_path(model_path: str) -> str:
    """Return local directory, resolving HuggingFace ids via the local cache."""
    if os.path.isdir(model_path):
        return model_path
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    slug = "models--" + model_path.replace("/", "--")
    snapshots = os.path.join(cache_dir, "hub", slug, "snapshots")
    if os.path.isdir(snapshots):
        snaps = sorted(Path(snapshots).iterdir(), key=lambda p: p.stat().st_mtime)
        if snaps:
            return str(snaps[-1])
    return model_path


def load_state_dict(model_path: str) -> Dict[str, torch.Tensor]:
    """Load weights, preferring safetensors shards. Falls back to HF AutoModel."""
    resolved = _resolve_model_path(model_path)
    sf_files = sorted(Path(resolved).glob("*.safetensors"))
    if sf_files:
        from safetensors.torch import load_file
        print(f"  Loading ({len(sf_files)} shards): {resolved}")
        sd: Dict[str, torch.Tensor] = {}
        for sf in sf_files:
            sd.update(load_file(str(sf), device="cpu"))
        return sd
    print(f"  Loading via HF: {model_path}")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32,
        device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True,
    )
    sd = model.state_dict()
    del model
    gc.collect()
    return sd


def save_model(
    base_model_path: str,
    merged_sd: Dict[str, torch.Tensor],
    out_dir: str,
    tokenizer_src: Optional[str] = None,
) -> None:
    """Save merged_sd as bfloat16 safetensors + copy config/tokenizer from base."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    resolved_base = _resolve_model_path(base_model_path)

    # Copy non-weight files (config, special_tokens_map, etc.); follow symlinks.
    for src in Path(resolved_base).iterdir():
        if src.is_dir():
            continue
        if src.suffix in {".safetensors", ".bin", ".pt"}:
            continue
        if src.name.endswith(".index.json"):
            continue
        real_src = Path(os.path.realpath(str(src)))
        dst = Path(out_dir) / src.name
        if not dst.exists():
            shutil.copy2(str(real_src), str(dst))

    # Tokenizer: prefer file copy, fall back to AutoTokenizer.save_pretrained.
    if not (Path(out_dir) / "tokenizer.json").exists():
        for tok_src in filter(None, [tokenizer_src, resolved_base, base_model_path]):
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
                tok.save_pretrained(out_dir)
                print(f"  Tokenizer → {out_dir}  (from {tok_src})")
                break
            except Exception as e:
                print(f"  WARNING: tokenizer copy failed ({tok_src}): {e}")
        else:
            raise RuntimeError(f"tokenizer copy failed for: {base_model_path}")

    weights_bf16 = {
        k: v.to(torch.bfloat16) if v.dtype in (torch.float32, torch.float16) else v
        for k, v in merged_sd.items()
        if isinstance(v, torch.Tensor)
    }
    out_path = os.path.join(out_dir, "model.safetensors")
    save_file(weights_bf16, out_path)
    print(f"  Saved {os.path.getsize(out_path)/1e9:.1f}GB → {out_dir}")


def get_2d_key_set(state_dict: Dict[str, torch.Tensor]) -> set:
    """2D float tensor keys — the matrices we run KT-Truncation/Polar on."""
    skip = {torch.int64, torch.uint8, torch.bool}
    return {k for k, v in state_dict.items() if v.ndim == 2 and v.dtype not in skip}


def compute_task_vector(
    base_sd: Dict[str, torch.Tensor],
    expert_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """τ_E = θ_E - θ_base (float32, numeric keys only)."""
    skip = {torch.int64, torch.uint8, torch.bool}
    return {
        k: expert_sd[k].float() - base_sd[k].float()
        for k in base_sd
        if k in expert_sd and base_sd[k].dtype not in skip
    }
