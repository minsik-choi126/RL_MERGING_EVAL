"""Stage A of proxy prep: download prompts (and gold answers when present).

Each Qwen3 RL dataset has a different schema:
    ifeval — prompt list, NO gold answer (RL uses rule-based verifier)
    math   — problem + answer (string)
    coding — responses_create_params.input + unit_tests, NO gold code answer
    lucy   — chat-format prompt list + completion (musique web-search agent
             RL data; gold completion present)

For ifeval/coding we save only the prompt; targets are generated later by the
task-specific expert in `generate_targets.py`. For math/lucy we save
(prompt, answer) directly using the gold answer/completion.

Output: data/training/{task}_raw.jsonl   ({"prompt": ..., "answer": ... or null})

Pre-validation split selection: prefer "validation"/"valid"/"val"/"dev"; fall
back to "train" with seeded random sampling.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

HF_DATASETS = {
    "ifeval": "nvidia/Nemotron-Cascade-RL-Instruction-Following",
    "math":   "nvidia/Nemotron-Cascade-RL-Math",
    "coding": "nvidia/Nemotron-RL-coding-competitive_coding",
    "lucy":   "Menlo/sft-lucy-musique-data-corrected-with-completion",
}


def _detect_split(hf_id: str) -> str:
    try:
        from datasets import get_dataset_split_names
        for cand in ("validation", "valid", "val", "dev"):
            if cand in set(get_dataset_split_names(hf_id)):
                return cand
    except Exception:
        pass
    return "train"


def _messages_to_prompt(msgs) -> str | None:
    """Extract a single user-message string from a list-of-dicts schema."""
    if not isinstance(msgs, list):
        return None
    parts = []
    for m in msgs:
        if not isinstance(m, dict): continue
        c = m.get("content")
        if isinstance(c, str) and c.strip():
            parts.append(c)
    return "\n\n".join(parts) if parts else None


def extract(task: str, row: dict) -> dict | None:
    """Return {'prompt': str, 'answer': str|None} or None if unsupported."""
    if task == "ifeval":
        prompt = _messages_to_prompt(row.get("prompt"))
        return {"prompt": prompt, "answer": None} if prompt else None

    if task == "math":
        prompt = row.get("problem") or row.get("prompt") or row.get("question")
        ans = row.get("answer") or row.get("solution")
        if prompt and ans:
            return {"prompt": str(prompt), "answer": str(ans)}
        return None

    if task == "coding":
        rcp = row.get("responses_create_params")
        if isinstance(rcp, dict):
            prompt = _messages_to_prompt(rcp.get("input"))
            return {"prompt": prompt, "answer": None} if prompt else None
        return None

    if task == "lucy":
        # Menlo/sft-lucy-musique-data-corrected-with-completion:
        # row['prompt']     = chat-format list  (system + user messages)
        # row['completion'] = chat-format list  (assistant turns; gold answer)
        prompt = _messages_to_prompt(row.get("prompt"))
        comp = row.get("completion")
        ans = _messages_to_prompt(comp) if comp is not None else None
        if not prompt:
            return None
        # final_answer is a short string sometimes; prefer the full completion text
        if not ans and isinstance(row.get("final_answer"), str):
            ans = str(row["final_answer"])
        return {"prompt": prompt, "answer": ans if ans else None}

    raise ValueError(f"unknown task {task}")


def download_one(task: str, hf_id: str, n: int, out_path: Path,
                 seed: int, max_scan: int):
    if out_path.exists():
        try:
            count = sum(1 for _ in open(out_path))
            if count >= n:
                print(f"[{task}] {out_path} already has {count} ≥ {n} rows → skip")
                return
        except Exception:
            pass

    split = _detect_split(hf_id)
    print(f"[{task}] streaming {hf_id} split={split} (target n={n}, seed={seed})")
    ds = load_dataset(hf_id, split=split, streaming=True)

    rng = np.random.default_rng(seed)
    pool = []
    seen = 0
    for row in ds:
        seen += 1
        rec = extract(task, row)
        if rec is None:
            if seen <= 3:
                print(f"  [warn] unsupported schema; row keys={list(row.keys())[:6]}")
            continue
        pool.append(rec)
        if len(pool) >= max_scan:
            break

    if len(pool) < n:
        raise RuntimeError(
            f"[{task}] only {len(pool)} usable rows in {hf_id}#{split} "
            f"after scanning {seen}. Lower n or expand max_scan."
        )

    idx = rng.permutation(len(pool))[:n]
    sampled = [pool[int(i)] for i in idx]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in sampled:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_with_ans = sum(1 for r in sampled if r["answer"])
    print(f"  → wrote {len(sampled)} rows to {out_path}  "
          f"(scanned={seen}, pool={len(pool)}, with_answer={n_with_ans})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_scan", type=int, default=20000)
    ap.add_argument("--out_dir",
                    default=str(Path(__file__).resolve().parent.parent / "data" / "training"))
    ap.add_argument("--tasks", nargs="+", default=list(HF_DATASETS))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for task in args.tasks:
        if task not in HF_DATASETS:
            raise ValueError(f"unknown task {task}")
        download_one(
            task, HF_DATASETS[task], args.n,
            out_dir / f"{task}_raw.jsonl", args.seed, args.max_scan,
        )

    print(f"\nDone. Raw inputs: {out_dir}/{{{','.join(args.tasks)}}}_raw.jsonl")
    print("Next: run generate_targets.py to fill in missing answers using task experts.")


if __name__ == "__main__":
    main()
