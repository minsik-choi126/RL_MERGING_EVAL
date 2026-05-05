"""W_col extraction (negative-direction, top-K Δlogp) with data-parallel
prompt splitting across DP=N GPUs.

For each expert (coding_rl, tool_rl, memory_rl):
    1. Load per-query records from data/per_query_rltrain/<task>.npz
    2. Build a token-level mask of "key" positions (bottom-K% Δlogp by default)
    3. Forward each prompt and accumulate
            ω_col[i, ℓ, c] = sum_t (mask[t] · |x_i,ℓ(t)[c]|)   /   n_key
       at every nn.Linear input via a forward hook (no clone — fused).

DP=N: prompts of each expert are sharded across N workers (one per GPU).
After each expert finishes, partial accumulators are summed.

Output:
    <out_npz>  with keys "<linear_name>.weight" → (3, d_in) float32
"""
from __future__ import annotations
import argparse, gc, os, pickle, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from helpers import resolve_path, setup_cache  # type: ignore

# Default data location; overridable via --data_dir.
DATA_DIR = THIS_DIR / "data" / "per_query_rltrain"

EXPERT_TASK = {"coding_rl": "coding", "tool_rl": "tool", "memory_rl": "memory"}
EXPERT_ORDER = ["coding_rl", "tool_rl", "memory_rl"]


# ─────────────────────────────────────────────────────────────────────────────
def load_task(task: str, expert: str, key_top_frac: float, select_mode: str,
              data_dir: Path):
    if not (0.0 < key_top_frac <= 1.0):
        raise ValueError(f"key_top_frac must be in (0, 1]; got {key_top_frac}")
    if select_mode not in ("top", "bottom", "bottom_complement", "top_complement"):
        raise ValueError(f"unknown select_mode: {select_mode}")
    z = np.load(data_dir / f"{task}.npz", allow_pickle=True)
    full_tokens = z["full_tokens"]
    full_seq_lens = [int(x) for x in z["full_seq_lens"]]
    prompt_lens = [int(x) for x in z["prompt_lens"]]
    ans_lens = [int(x) for x in z["seq_lens"]]
    en = [str(n) for n in z["expert_names"]]
    base_lp = z["base_lp"].astype(np.float32)
    exp_lp = z["expert_lp"].astype(np.float32)[en.index(expert)]
    delta = exp_lp - base_lp

    n_total = len(delta)
    n_key_target = int(np.ceil(key_top_frac * n_total))
    if select_mode in ("top", "top_complement"):
        tail_idx = np.argpartition(-delta, n_key_target - 1)[:n_key_target]
    else:
        tail_idx = np.argpartition(delta, n_key_target - 1)[:n_key_target]
    tail_mask = np.zeros(n_total, dtype=np.float32)
    tail_mask[tail_idx] = 1.0
    if select_mode.endswith("_complement"):
        key_mask = 1.0 - tail_mask
    else:
        key_mask = tail_mask

    seqs, pls, masks = [], [], []
    off_full, off_ans = 0, 0
    for pl, flen, alen in zip(prompt_lens, full_seq_lens, ans_lens):
        flen, alen = int(flen), int(alen)
        seq = torch.from_numpy(full_tokens[off_full: off_full + flen].astype(np.int64))
        m = torch.from_numpy(key_mask[off_ans: off_ans + alen].copy())
        seqs.append(seq); pls.append(pl); masks.append(m)
        off_full += flen; off_ans += alen
    return seqs, pls, masks


# ─────────────────────────────────────────────────────────────────────────────
def worker(expert_idx: int, expert_name: str, expert_path: str, device: str,
           key_top_frac: float, select_mode: str, max_prompts,
           result_path: str, prompt_slice, data_dir_str: str):
    """Process a slice of one expert's prompts on a single GPU."""
    torch.cuda.set_device(device)
    data_dir = Path(data_dir_str)
    tag = f"[w{expert_idx}/{expert_name}|{prompt_slice[0]}:{prompt_slice[1]}]"
    print(f"{tag} start on {device}", flush=True)

    t0 = time.time()
    task = EXPERT_TASK[expert_name]
    seqs, pls, masks = load_task(task, expert_name, key_top_frac, select_mode, data_dir)
    if max_prompts:
        seqs = seqs[:max_prompts]; pls = pls[:max_prompts]; masks = masks[:max_prompts]
    s, e = prompt_slice
    seqs = seqs[s:e]; pls = pls[s:e]; masks = masks[s:e]
    print(f"{tag} {len(seqs)} prompts, load={time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    expert = AutoModelForCausalLM.from_pretrained(
        resolve_path(expert_path), torch_dtype=torch.bfloat16,
        device_map={"": device},
    ).eval()
    print(f"{tag} model loaded in {time.time()-t1:.1f}s", flush=True)

    # Fused-accumulator hook: |x|*mask -> sum over time -> add into per-layer running sum.
    state = {"a": 0, "b": 0, "mask": None}
    acc_dev: dict = {}
    handles, linear_names = [], []
    for name, mod in expert.named_modules():
        if isinstance(mod, nn.Linear):
            linear_names.append(name)
            def make_hook(nm):
                def hook(_m, inp, _out):
                    x = inp[0]
                    a, b = state["a"], state["b"]
                    if x.dim() == 3:
                        x_seg = x[0, a:b]
                    elif x.dim() == 2:
                        x_seg = x[a:b]
                    else:
                        return
                    contrib = (x_seg.abs().float() * state["mask"][:, None]).sum(dim=0)
                    if nm in acc_dev:
                        acc_dev[nm].add_(contrib)
                    else:
                        acc_dev[nm] = contrib
                return hook
            handles.append(mod.register_forward_hook(make_hook(name)))

    n_key_total = 0
    t_loop = time.time()
    with torch.no_grad():
        for qi, (seq, pl, km) in enumerate(zip(seqs, pls, masks), 1):
            Lseq = int(seq.shape[0])
            if Lseq <= pl:
                continue
            state["a"] = pl - 1
            state["b"] = Lseq - 1
            state["mask"] = km.to(device=device, dtype=torch.float32, non_blocking=True)
            n_key_total += int(state["mask"].sum().item())
            _ = expert(seq.unsqueeze(0).to(device, non_blocking=True), use_cache=False)
            if qi % 20 == 0 or qi == len(seqs):
                dt = time.time() - t_loop
                eta = dt / qi * (len(seqs) - qi)
                print(f"{tag} {qi}/{len(seqs)} ({dt:.0f}s, ETA {eta:.0f}s)", flush=True)

    for h in handles:
        h.remove()

    acc_cpu = {f"xabs_key::{nm}.weight": acc_dev[nm].cpu()
               for nm in linear_names if nm in acc_dev}
    with open(result_path, "wb") as f:
        pickle.dump({"acc_c": acc_cpu, "n_key": n_key_total,
                       "linear_names": linear_names, "wall": time.time() - t0}, f)
    print(f"{tag} DONE wall={time.time()-t0:.1f}s", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
def merge_partial_pickles(partial_paths: list, out_path: str):
    """Aggregate per-shard accumulators (sum) and total n_key across shards."""
    combined_acc = None
    n_key_total = 0
    linear_names = None
    wall_max = 0.0
    for p in partial_paths:
        with open(p, "rb") as f:
            r = pickle.load(f)
        if combined_acc is None:
            combined_acc = {k: v.clone() for k, v in r["acc_c"].items()}
            linear_names = r["linear_names"]
        else:
            for k, v in r["acc_c"].items():
                if k in combined_acc:
                    combined_acc[k].add_(v)
                else:
                    combined_acc[k] = v.clone()
        n_key_total += r["n_key"]
        wall_max = max(wall_max, r.get("wall", 0.0))
    with open(out_path, "wb") as f:
        pickle.dump({"acc_c": combined_acc, "n_key": n_key_total,
                       "linear_names": linear_names, "wall": wall_max}, f)


def main():
    ap = argparse.ArgumentParser(description="W_col extraction with DP=N prompt sharding")
    ap.add_argument("--base_model", required=True,
                     help="Base model HF id or local path (only used for shape sanity)")
    ap.add_argument("--expert_coding", required=True,
                     help="ReasonFlux-Coder-7B path or HF id")
    ap.add_argument("--expert_tool", required=True,
                     help="Qwen2.5-7B-Instruct-ToolRL-grpo-cold path or HF id")
    ap.add_argument("--expert_memory", required=True,
                     help="RL-MemoryAgent-7B path or HF id")
    ap.add_argument("--out_npz", required=True,
                     help="output W npz file")
    ap.add_argument("--dp", type=int, default=2,
                     help="data-parallel degree N (number of GPUs)")
    ap.add_argument("--key_top_frac", type=float, default=0.10,
                     help="per-expert key fraction by Δlogp (default 0.10)")
    ap.add_argument("--select_mode", default="bottom",
                     choices=["top", "bottom", "bottom_complement", "top_complement"],
                     help="default 'bottom' = negative-direction (suppression) variant")
    ap.add_argument("--data_dir", default=str(DATA_DIR),
                     help="dir with per-task npz files (coding/tool/memory)")
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--cache_dir", default=None,
                     help="optional HF cache dir override")
    args = ap.parse_args()

    setup_cache(args.cache_dir)
    if args.dp < 1:
        raise ValueError("--dp must be >= 1")
    n_gpus = torch.cuda.device_count()
    if n_gpus < args.dp:
        raise RuntimeError(f"--dp={args.dp} but only {n_gpus} CUDA device(s) visible")

    devices = [f"cuda:{i}" for i in range(args.dp)]
    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / f"_tmp_{out_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    expert_paths = {
        "coding_rl": args.expert_coding,
        "tool_rl":   args.expert_tool,
        "memory_rl": args.expert_memory,
    }

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print("=" * 64)
    print(f"  W_col extraction  DP={args.dp}   start={time.strftime('%F %T')}")
    print(f"  GPUs: {devices}")
    print(f"  experts: {EXPERT_ORDER}")
    print(f"  select_mode={args.select_mode}  key_top_frac={args.key_top_frac}")
    print(f"  out_npz={out_path}")
    print("=" * 64)

    t_total_start = time.time()

    final_paths = []
    for ei, expert_name in enumerate(EXPERT_ORDER):
        # determine prompt count for this task
        z = np.load(Path(args.data_dir) / f"{EXPERT_TASK[expert_name]}.npz", allow_pickle=True)
        n_total = len(z["prompt_lens"])
        if args.max_prompts:
            n_total = min(n_total, args.max_prompts)
        del z

        # split into N (= dp) chunks
        N = args.dp
        chunk_starts = [(n_total * i) // N for i in range(N + 1)]
        slices = [(chunk_starts[i], chunk_starts[i + 1]) for i in range(N)]
        slices = [(s, e) for s, e in slices if s < e]

        partial_paths = [str(tmp_dir / f"{expert_name}_part{i}.pkl")
                          for i in range(len(slices))]
        print(f"\n[expert {ei+1}/3] {expert_name}: prompts={n_total}, "
              f"split={[s for s in slices]}")
        t_e = time.time()
        procs = []
        for i, (sl, dev) in enumerate(zip(slices, devices)):
            p = mp.Process(target=worker, args=(ei, expert_name, expert_paths[expert_name],
                                                  dev, args.key_top_frac, args.select_mode,
                                                  args.max_prompts, partial_paths[i],
                                                  sl, args.data_dir))
            p.start(); procs.append(p)
        for p in procs:
            p.join()
        for i, p in enumerate(procs):
            if p.exitcode != 0:
                print(f"[FATAL] expert {expert_name} shard {i} exit={p.exitcode}")
                sys.exit(1)
        # combine shards into one expert pickle
        expert_pkl = str(tmp_dir / f"{expert_name}.pkl")
        merge_partial_pickles(partial_paths, expert_pkl)
        final_paths.append(expert_pkl)
        print(f"[expert {ei+1}/3] {expert_name} done in {time.time()-t_e:.1f}s")

    t_extraction = time.time() - t_total_start

    # ── Aggregate all experts → npz ──
    results = []
    for path in final_paths:
        with open(path, "rb") as f:
            results.append(pickle.load(f))

    union_layers = set().union(*[set(r["acc_c"].keys()) for r in results])
    layer_keys = sorted({k.split("::", 1)[1] for k in union_layers
                          if k.startswith("xabs_key::")})

    payload = {}
    for layer_key in layer_keys:
        stack = []
        ok = True
        for r in results:
            k = f"xabs_key::{layer_key}"
            if k not in r["acc_c"]:
                ok = False; break
            arr = r["acc_c"][k].numpy() / max(r["n_key"], 1)
            stack.append(arr)
        if not ok or len(stack) != 3:
            continue
        if len({a.shape[0] for a in stack}) != 1:
            continue
        payload[layer_key] = np.stack(stack, axis=0).astype(np.float32)

    np.savez_compressed(out_path, **payload)

    print()
    print("=" * 64)
    print(f"  TOTAL extraction wall = {t_extraction:.1f}s ({t_extraction/60:.2f}min)")
    print(f"  Per-expert worker walls (max across DP shards):")
    for r, name in zip(results, EXPERT_ORDER):
        print(f"    {name}: {r['wall']:.1f}s  (n_key={r['n_key']})")
    print(f"  Output: {out_path}  ({out_path.stat().st_size/1e6:.1f}MB, "
          f"{len(payload)} layers)")
    print("=" * 64)

    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
