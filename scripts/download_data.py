#!/usr/bin/env python3
"""
Download and stage every dataset the eval pipeline needs.

Sources:
  aime24 — Maxwell-Jia/AIME_2024
  aime25 — yentinglin/aime_2025
  aime26 — opencompass/AIME2025       (placeholder; swap when AIME-2026 is released)
  IFEval — google/IFEval               (loaded on the fly by ifeval_eval.py — no action)
  MemAgent RULER-HQA — BytedTsinghua-SIA/MemAgent-Eval (Bytedance's release)

For AIME, we build "VERL-ready" parquets with the prompt + boxed-answer reward
shape that verl.trainer.main_ppo expects. Files written:
  $EVAL_DATA_ROOT/aime24/test_verl_ready_with_instruction.parquet
  $EVAL_DATA_ROOT/aime25/test_verl_ready_with_instruction.parquet
  $EVAL_DATA_ROOT/aime26/test_verl_ready_with_instruction.parquet
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

AIME_SOURCES = {
    "aime24": ("Maxwell-Jia/AIME_2024", "train"),
    "aime25": ("yentinglin/aime_2025", "train"),
    # AIME 2026 is not on HF yet — reuse 2025 as a placeholder so the pipeline
    # does not error. Overwrite when the official release lands.
    "aime26": ("yentinglin/aime_2025", "train"),
}


def _detect_columns(row):
    """AIME datasets use slightly different column names; normalize them."""
    problem = row.get("Problem") or row.get("problem") or row.get("question")
    answer = row.get("Answer") or row.get("answer")
    if problem is None or answer is None:
        raise KeyError(f"could not find problem/answer in row keys {list(row)}")
    return str(problem).strip(), str(answer).strip()


def build_aime_parquets(data_root: Path, variants: list[str], force: bool) -> None:
    import pandas as pd
    from datasets import load_dataset

    for variant in variants:
        if variant not in AIME_SOURCES:
            print(f"  [warn] unknown variant {variant!r}, skipping")
            continue
        hf_name, split = AIME_SOURCES[variant]
        out_dir = data_root / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "test_verl_ready_with_instruction.parquet"
        if out_file.exists() and not force:
            print(f"  [{variant}] already present → {out_file}")
            continue

        print(f"  [{variant}] loading {hf_name} split={split} ...")
        ds = load_dataset(hf_name, split=split)

        rows = []
        for idx, row in enumerate(ds):
            problem, answer = _detect_columns(row)
            user_content = f"{problem}\n\n{INSTRUCTION}"
            rows.append({
                "data_source": "aime",
                "prompt": [{"role": "user", "content": user_content}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": "test",
                    "index": idx,
                    "variant": variant,
                    "source": hf_name,
                },
            })
        df = pd.DataFrame(rows)
        df.to_parquet(out_file, index=False)
        print(f"  [{variant}] wrote {len(df)} rows → {out_file}")


def download_memagent_hqa(memagent_root: Path, force: bool) -> None:
    """Pull the RULER-HotpotQA eval JSONs used by MemAgent/ruler_hqa.py."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  [memagent] huggingface_hub not installed; skip")
        return

    dest = memagent_root / "hotpotqa"
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "eval_50.json"
    if marker.exists() and not force:
        print(f"  [memagent] already present → {dest}")
        return

    repo_id = os.environ.get("MEMAGENT_HQA_REPO", "BytedTsinghua-SIA/hotpotqa")
    print(f"  [memagent] downloading {repo_id} → {dest}")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest),
            allow_patterns=["eval_*.json", "hotpotqa_dev.parquet", "README.md"],
        )
    except Exception as e:
        print(f"  [memagent] download failed ({e}). "
              f"You can manually drop eval_*.json files into {dest}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", default=os.environ.get("EVAL_DATA_ROOT", "data"),
        help="Where AIME parquets are written (default: $EVAL_DATA_ROOT or ./data)",
    )
    parser.add_argument(
        "--memagent-root", default=str(Path(__file__).resolve().parent.parent / "MemAgent"),
        help="MemAgent directory (default: <repo>/MemAgent)",
    )
    parser.add_argument(
        "--variants", nargs="+", default=["aime24", "aime25", "aime26"],
        help="Which AIME variants to fetch",
    )
    parser.add_argument(
        "--skip-aime", action="store_true", help="Skip AIME parquet generation",
    )
    parser.add_argument(
        "--skip-memagent", action="store_true", help="Skip MemAgent hotpotqa download",
    )
    parser.add_argument("--force", action="store_true", help="Re-download / overwrite")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    memagent_root = Path(args.memagent_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Downloading evaluation datasets")
    print("=" * 64)
    print(f"  data_root    : {data_root}")
    print(f"  memagent_root: {memagent_root}")

    if not args.skip_aime:
        try:
            build_aime_parquets(data_root, args.variants, args.force)
        except Exception as e:
            print(f"  [aime] ERROR: {e}")
            return 1
    if not args.skip_memagent:
        download_memagent_hqa(memagent_root, args.force)

    print("=" * 64)
    print("  Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
