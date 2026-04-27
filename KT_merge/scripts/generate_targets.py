"""Stage B of proxy prep: fill in missing 'answer' fields using task-specific experts.

For each task's _raw.jsonl, if a row's "answer" is null, generate a target by
running the task-specific RL expert on the prompt (greedy, max_new_tokens N).
Writes the completed file to {task}.jsonl.

Why task-specific expert? The target should be a sequence the expert assigns
high probability to (its own preferred response). This makes Δ_E > 0 by
construction at most positions for the matching expert, which is exactly the
"key event" signal we want — it captures which positions encode that expert's
specialized capability.

For math, gold answers are usually present and used as-is.

Usage:
    python generate_targets.py \\
        --ifeval models/ifeval --math models/math --coding models/coding \\
        --device cuda:0 --max_new_tokens 512
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_raw(path: Path) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            out.append(json.loads(line))
    return out


def save(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@torch.no_grad()
def generate_for_task(task: str, expert_path: str, rows: List[dict],
                      tokenizer, device: str, max_new_tokens: int,
                      max_prompt_tokens: int = 8192) -> List[dict]:
    needs_gen = [i for i, r in enumerate(rows) if not r.get("answer")]
    if not needs_gen:
        print(f"[{task}] all {len(rows)} rows already have answers — skip generation")
        return rows

    print(f"[{task}] loading expert: {expert_path}")
    model = AutoModelForCausalLM.from_pretrained(
        expert_path, torch_dtype=torch.bfloat16,
        device_map={"": device}, low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"[{task}] generating answers for {len(needs_gen)}/{len(rows)} rows "
          f"(max_new_tokens={max_new_tokens})")
    for n_done, idx in enumerate(needs_gen, start=1):
        prompt = rows[idx]["prompt"]
        try:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=True,
            )
        except Exception:
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ids = ids[:max_prompt_tokens]
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            ids_t,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[0, len(ids):].tolist()
        gen = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        rows[idx]["answer"] = gen
        if n_done <= 3 or n_done % 16 == 0 or n_done == len(needs_gen):
            print(f"  [{n_done}/{len(needs_gen)}] gen_len={len(gen_ids)}  "
                  f"first 80 chars: {gen[:80]!r}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ifeval", required=True)
    ap.add_argument("--math",   required=True)
    ap.add_argument("--coding", required=True)
    ap.add_argument("--data_dir",
                    default=str(Path(__file__).resolve().parent.parent / "data" / "training"))
    ap.add_argument("--tokenizer_src", default="Qwen/Qwen3-1.7B",
                    help="tokenizer source (base model id, shared by all experts)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--tasks", nargs="+", default=["ifeval", "math", "coding"])
    args = ap.parse_args()

    expert_paths: Dict[str, str] = {
        "ifeval": args.ifeval, "math": args.math, "coding": args.coding,
    }

    print(f"[init] loading tokenizer: {args.tokenizer_src}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_src,
                                               trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_dir = Path(args.data_dir)
    for task in args.tasks:
        raw_path = data_dir / f"{task}_raw.jsonl"
        out_path = data_dir / f"{task}.jsonl"
        if not raw_path.exists():
            raise FileNotFoundError(f"{raw_path} missing — run download_training_data.py first")
        if out_path.exists():
            print(f"[{task}] {out_path} already exists → skip")
            continue
        rows = load_raw(raw_path)
        rows = generate_for_task(
            task, expert_paths[task], rows, tokenizer,
            args.device, args.max_new_tokens,
        )
        save(out_path, rows)
        n_with = sum(1 for r in rows if r.get("answer"))
        print(f"[{task}] → {out_path}  ({n_with}/{len(rows)} have answers)")


if __name__ == "__main__":
    main()
