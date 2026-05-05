"""Shared model-IO + path helpers (self-contained, no extra deps)."""
from __future__ import annotations
import gc, os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_cache(cache_dir: Optional[str] = None):
    """Set HF cache dir (no-op if not provided / not creatable)."""
    if cache_dir:
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "hub"))


def resolve_path(model_path: str) -> str:
    """Local dir as-is; otherwise look up the most-recent HF cache snapshot."""
    if os.path.isdir(model_path):
        return model_path
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    slug = "models--" + model_path.replace("/", "--")
    snapshots = os.path.join(hf_home, "hub", slug, "snapshots")
    if os.path.isdir(snapshots):
        snaps = sorted(Path(snapshots).iterdir(), key=lambda p: p.stat().st_mtime)
        if snaps:
            return str(snaps[-1])
    return model_path


def load_state_dict(model_path: str) -> dict:
    """Load state_dict on CPU. Prefer safetensors; fall back to HF AutoModel."""
    resolved = resolve_path(model_path)
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
    """Save merged state dict in HF format (safetensors + tokenizer)."""
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


def get_numeric_keys(base_sd: dict, expert_sds: list) -> list:
    common = set(base_sd.keys())
    for sd in expert_sds:
        common &= set(sd.keys())
    return sorted(
        k for k in common
        if base_sd[k].dtype not in (torch.int64, torch.uint8)
    )
