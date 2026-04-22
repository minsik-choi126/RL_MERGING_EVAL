#!/usr/bin/env python3
"""
Unified evaluation orchestrator for all benchmarks.

This is a THIN orchestrator that delegates to existing evaluation scripts
while providing consistent parameter handling, model auto-detection, and
unified result collection.

Supported benchmarks:
    aime24, aime25, aime26 - AIME math (VERL framework)
    ifeval            - Instruction Following (vLLM direct)
    livebench         - LiveBench coding (vLLM multi-engine)
    livecodebench     - LiveCodeBench coding (vLLM multi-engine)
    coding            - LiveBench + LiveCodeBench (alias)
    tool_use          - BFCL function calling (vLLM backend)
    memagent          - RULER HQA long-context memory (vLLM serve)

Usage:
    python unified_eval.py --model /path/to/model --benchmarks all
    python unified_eval.py --model /path/to/model --benchmarks aime24 aime25 aime26 ifeval
    python unified_eval.py --model /path/to/model --benchmarks coding --gpu_per_engine 4
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================================
# Configuration
# ============================================================================

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
EVAL_SCRIPTS_DIR = EVAL_DIR / "eval_scripts"

ALL_BENCHMARKS = [
    "aime24", "aime25", "aime26", "ifeval",
    "livebench", "livecodebench",
    "tool_use", "memagent",
]

# Aliases: "coding" expands to livebench + livecodebench, "all" expands to everything
BENCHMARK_ALIASES = {
    "all": ALL_BENCHMARKS,
    "coding": ["livebench", "livecodebench"],
}


@dataclass
class ModelInfo:
    """Auto-detected model properties."""
    path: str
    short_name: str = ""
    max_position_embeddings: int = 32768
    has_rope_scaling: bool = False
    rope_scaling_type: str = ""
    rope_scaling_factor: float = 1.0
    model_type: str = ""
    num_attention_heads: int = 0
    num_key_value_heads: int = 0

    def __post_init__(self):
        if not self.short_name:
            self.short_name = Path(self.path.rstrip("/")).name


@dataclass
class EvalConfig:
    """Unified evaluation configuration."""
    model: str = ""
    model_info: Optional[ModelInfo] = None
    benchmarks: List[str] = field(default_factory=list)
    tp: int = 4
    n_gpus: int = 0  # 0 = auto-detect
    gpu_per_engine: int = 1
    output_dir: str = ""
    log_dir: str = ""  # directory for per-benchmark logs
    run_num: int = 0  # 0 = auto-increment
    temperature: Optional[float] = None
    chat_template: str = ""
    memagent_tests: str = "hqa"
    memagent_method: str = "openai"
    dry_run: bool = False
    verbose: bool = False


# ============================================================================
# Model Auto-Detection
# ============================================================================

def detect_model_info(model_path: str) -> ModelInfo:
    """Read config.json from model path to detect properties."""
    info = ModelInfo(path=model_path)
    config_path = Path(model_path) / "config.json"

    if not config_path.exists():
        print(f"  [warn] config.json not found at {config_path}, using defaults")
        return info

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [warn] Failed to read config.json: {e}")
        return info

    info.max_position_embeddings = cfg.get("max_position_embeddings", 32768)
    info.model_type = cfg.get("model_type", "")
    info.num_attention_heads = cfg.get("num_attention_heads", 0)
    info.num_key_value_heads = cfg.get("num_key_value_heads", 0)

    rope = cfg.get("rope_scaling")
    if rope and isinstance(rope, dict):
        info.has_rope_scaling = True
        info.rope_scaling_type = rope.get("type", rope.get("rope_type", ""))
        info.rope_scaling_factor = rope.get("factor", 1.0)

    print(f"  [model] max_position_embeddings={info.max_position_embeddings}")
    print(f"  [model] model_type={info.model_type}")
    if info.has_rope_scaling:
        print(f"  [model] rope_scaling: type={info.rope_scaling_type}, "
              f"factor={info.rope_scaling_factor}")
    if info.num_key_value_heads:
        print(f"  [model] heads={info.num_attention_heads}, "
              f"kv_heads={info.num_key_value_heads}")

    return info


def detect_gpu_count() -> int:
    """Detect available GPU count via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            count = len([l for l in result.stdout.strip().split("\n") if l.strip()])
            return max(count, 1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    print("  [warn] nvidia-smi not available, defaulting to 4 GPUs")
    return 4


# ============================================================================
# Run Number Management
# ============================================================================

def determine_run_num(results_dir: Path) -> int:
    """Auto-increment run number based on existing run_NNN directories."""
    if not results_dir.exists():
        return 1
    existing = []
    for d in results_dir.iterdir():
        m = re.match(r"^run_(\d+)$", d.name)
        if m and d.is_dir():
            existing.append(int(m.group(1)))
    return max(existing) + 1 if existing else 1


# ============================================================================
# Utility: Run Shell Command
# ============================================================================

def run_command(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    dry_run: bool = False,
    label: str = "",
    log_file: Optional[str] = None,
) -> int:
    """Run a shell command with real-time output streaming and optional log file."""
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
    if label:
        print(f"\n  [{label}] Running: {cmd_str}")
    else:
        print(f"\n  Running: {cmd_str}")

    if dry_run:
        print("  [dry-run] Skipped")
        return 0

    merged_env = {**os.environ}
    if env:
        merged_env.update(env)

    log_fh = None
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_file, "a", encoding="utf-8")

    try:
        proc = subprocess.Popen(
            cmd_str, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=merged_env, cwd=cwd,
            bufsize=1, universal_newlines=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_fh:
                log_fh.write(line)
        proc.wait()
        return proc.returncode
    except Exception as e:
        print(f"  [ERROR] Command failed: {e}")
        return 1
    finally:
        if log_fh:
            log_fh.close()


def cleanup_ray():
    """Stop stale Ray processes to avoid NFS .nfs silly-rename issues."""
    subprocess.run(
        "ray stop --force 2>/dev/null; sleep 2",
        shell=True, capture_output=True,
    )


# ============================================================================
# Benchmark Runners
# ============================================================================

def run_aime(config: EvalConfig, variant: str) -> dict:
    """Run AIME24, AIME25, or AIME26 evaluation via VERL.

    Args:
        variant: "aime24", "aime25", or "aime26"
    """
    cleanup_ray()

    info = config.model_info
    max_pos = info.max_position_embeddings if info else 32768
    # Auto-adjust max_response_length: min(30720, max_pos - 2048)
    max_response_length = min(30720, max_pos - 2048)
    if max_response_length < 1024:
        print(f"  [warn] max_response_length={max_response_length} is very small "
              f"(max_pos={max_pos}). Setting to 1024.")
        max_response_length = 1024

    print(f"  [aime] max_position_embeddings={max_pos}, "
          f"max_response_length={max_response_length}")

    script = EVAL_SCRIPTS_DIR / f"run_eval_{variant}.sh"
    if not script.exists():
        return {"status": "error", "message": f"Script not found: {script}"}

    env_overrides = {
        "MODEL_PATH": config.model,
        "N_GPUS_PER_NODE": str(config.n_gpus),
        "MAX_RESPONSE_LENGTH": str(max_response_length),
    }

    log_file = str(Path(config.log_dir) / f"{variant}.log") if config.log_dir else None

    rc = run_command(
        [f"bash {script}"],
        env=env_overrides,
        dry_run=config.dry_run,
        label=variant,
        log_file=log_file,
    )

    return {
        "status": "success" if rc == 0 else "error",
        "return_code": rc,
        "max_response_length": max_response_length,
    }


def run_ifeval(config: EvalConfig) -> dict:
    """Run IFEval evaluation via vLLM direct inference."""
    script = EVAL_SCRIPTS_DIR / "run_eval_ifeval.sh"
    if not script.exists():
        return {"status": "error", "message": f"Script not found: {script}"}

    env_overrides = {
        "MODEL_PATH": config.model,
        "TP": str(config.tp),
    }

    log_file = str(Path(config.log_dir) / "ifeval.log") if config.log_dir else None

    rc = run_command(
        [f"bash {script}"],
        env=env_overrides,
        dry_run=config.dry_run,
        label="ifeval",
        log_file=log_file,
    )

    return {"status": "success" if rc == 0 else "error", "return_code": rc}


def run_coding(config: EvalConfig, dataset: str) -> dict:
    """Run LiveBench or LiveCodeBench evaluation.

    Args:
        dataset: "LiveBench" or "LiveCodeBench"
    """
    script = EVAL_DIR / "Coding" / "run_eval.sh"
    if not script.exists():
        return {"status": "error", "message": f"Script not found: {script}"}

    temp_arg = str(config.temperature) if config.temperature is not None else ""
    chat_arg = config.chat_template or ""

    cmd = (
        f"bash {script} "
        f'"{config.model}" '
        f'"{dataset}" '
        f'"{config.gpu_per_engine}" '
        f'"" '  # total_gpus (auto-detected inside script)
        f'"{temp_arg}" '
        f'"{chat_arg}"'
    )

    log_file = str(Path(config.log_dir) / f"{dataset.lower()}.log") if config.log_dir else None

    rc = run_command(
        [cmd],
        dry_run=config.dry_run,
        label=dataset.lower(),
        log_file=log_file,
    )

    return {"status": "success" if rc == 0 else "error", "return_code": rc}


def run_tool_use(config: EvalConfig) -> dict:
    """Run BFCL Tool Use evaluation."""
    script = EVAL_DIR / "Tool_use" / "run_eval.sh"
    if not script.exists():
        return {"status": "error", "message": f"Script not found: {script}"}

    cmd = f'bash {script} "{config.model}" "{config.tp}"'
    log_file = str(Path(config.log_dir) / "tool_use.log") if config.log_dir else None

    rc = run_command(
        [cmd],
        dry_run=config.dry_run,
        label="tool_use",
        log_file=log_file,
    )

    return {"status": "success" if rc == 0 else "error", "return_code": rc}


def run_memagent(config: EvalConfig) -> dict:
    """Run MemAgent RULER HQA evaluation."""
    script = EVAL_DIR / "MemAgent" / "run_eval.sh"
    if not script.exists():
        return {"status": "error", "message": f"Script not found: {script}"}

    cmd = (
        f'bash {script} '
        f'"{config.model}" '
        f'--tp {config.tp} '
        f'--method {config.memagent_method} '
        f'--tests {config.memagent_tests}'
    )

    log_file = str(Path(config.log_dir) / "memagent.log") if config.log_dir else None

    rc = run_command(
        [cmd],
        dry_run=config.dry_run,
        label="memagent",
        log_file=log_file,
    )

    return {"status": "success" if rc == 0 else "error", "return_code": rc}


# ============================================================================
# Result Collection
# ============================================================================

def collect_coding_result(config: EvalConfig, run_dir: Path, dataset: str):
    """Copy coding results into the run directory."""
    model_short = config.model_info.short_name
    coding_result_dir = EVAL_DIR / "Coding" / "evaluation" / "results"
    result_file = coding_result_dir / f"results-eval-{model_short}-{dataset}.txt"

    combined_dir = run_dir / "Coding"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_file = combined_dir / f"{model_short}.txt"

    if result_file.exists():
        with open(combined_file, "a") as out:
            out.write(f"\n=== {dataset} ===\n")
            out.write(result_file.read_text())
        print(f"  -> Saved: {combined_file} ({dataset})")
    else:
        print(f"  [warn] Result file not found: {result_file}")


def _print_bfcl_table(model_short: str, score_dir: Path):
    """Print BFCL results using the shared print_bfcl_table script."""
    script = EVAL_DIR / "Tool_use" / "print_bfcl_table.py"
    try:
        subprocess.run(
            [sys.executable, str(script), model_short,
             "--score-dir", str(score_dir)],
            check=False,
        )
    except Exception as e:
        print(f"  [warn] Failed to print BFCL table: {e}")


def collect_tool_use_result(config: EvalConfig, run_dir: Path):
    """Copy BFCL score results into the run directory and print summary table."""
    model_short = config.model_info.short_name
    bfcl_score_dir = (
        EVAL_DIR / "Tool_use" / "berkeley-function-call-leaderboard"
        / "score" / model_short
    )

    if bfcl_score_dir.exists():
        dest = run_dir / "Tool_use" / model_short
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(bfcl_score_dir, dest)
        print(f"  -> Saved: {dest}/")

        # Print table
        _print_bfcl_table(model_short, bfcl_score_dir)
    else:
        print(f"  [warn] BFCL score dir not found: {bfcl_score_dir}")


def collect_memagent_result(config: EvalConfig, run_dir: Path):
    """Copy MemAgent results into the run directory."""
    model_short = config.model_info.short_name
    memagent_base = EVAL_DIR / "MemAgent" / "taskutils" / "memory_eval" / "results"

    if not memagent_base.exists():
        print("  [warn] MemAgent results dir not found")
        return

    found = False
    for dist_dir in memagent_base.iterdir():
        if not dist_dir.is_dir() or not dist_dir.name.startswith("ruler_hqa_"):
            continue
        jsonl = dist_dir / f"{model_short}.jsonl"
        if jsonl.exists():
            dest_dir = run_dir / "MemAgent" / dist_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(jsonl, dest_dir / jsonl.name)
            print(f"  -> Saved: {dest_dir}/{jsonl.name}")
            found = True

    if not found:
        print("  [warn] No MemAgent result files found")


# ============================================================================
# Summary
# ============================================================================

def print_summary(results: Dict[str, dict], elapsed: float, run_dir: Path):
    """Print a summary table of all benchmark results."""
    print()
    print("=" * 64)
    print("  Evaluation Summary")
    print("=" * 64)
    print(f"  {'Benchmark':<20} {'Status':<10} {'Return Code':<15} {'Notes'}")
    print(f"  {'-'*20} {'-'*10} {'-'*15} {'-'*20}")

    for bench, result in results.items():
        status = result.get("status", "unknown")
        rc = result.get("return_code", "N/A")
        msg = result.get("message", "")
        notes = msg if status == "error" and msg else ""
        if "max_response_length" in result:
            mrl = f"max_resp_len={result['max_response_length']}"
            notes = f"{notes} {mrl}".strip() if notes else mrl
        status_str = "OK" if status == "success" else "FAIL"
        print(f"  {bench:<20} {status_str:<10} {str(rc):<15} {notes}")

    # Timing
    mins, secs = divmod(int(elapsed), 60)
    hours, mins = divmod(mins, 60)
    print(f"\n  Total time: {hours}h {mins}m {secs}s")
    print(f"  Results:    {run_dir}/")
    print("=" * 64)

    # Write summary JSON
    summary = {
        "benchmarks": results,
        "elapsed_seconds": round(elapsed, 1),
        "run_dir": str(run_dir),
    }
    summary_path = run_dir / "summary.json"
    try:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary written to: {summary_path}")
    except OSError as e:
        print(f"  [warn] Could not write summary: {e}")


# ============================================================================
# Main Orchestrator
# ============================================================================

def resolve_benchmarks(benchmark_args: List[str]) -> List[str]:
    """Expand aliases and deduplicate benchmark list."""
    resolved = []
    for b in benchmark_args:
        b_lower = b.lower().strip()
        if b_lower in BENCHMARK_ALIASES:
            resolved.extend(BENCHMARK_ALIASES[b_lower])
        elif b_lower in ALL_BENCHMARKS:
            resolved.append(b_lower)
        else:
            print(f"  [warn] Unknown benchmark '{b}', skipping. "
                  f"Valid: {', '.join(ALL_BENCHMARKS + list(BENCHMARK_ALIASES.keys()))}")
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for b in resolved:
        if b not in seen:
            seen.add(b)
            deduped.append(b)
    return deduped


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unified_eval.py --model /path/to/model --benchmarks all
  python unified_eval.py --model /path/to/model --benchmarks aime24 aime25
  python unified_eval.py --model /path/to/model --benchmarks coding --gpu_per_engine 4
  python unified_eval.py --model /path/to/model --benchmarks ifeval --tp 4
  python unified_eval.py --model /path/to/model --benchmarks memagent --tp 1 --memagent_tests hqa
        """,
    )
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument(
        "--benchmarks", nargs="+", default=["all"],
        help="Benchmarks to run (space-separated). "
             "Options: all, coding, aime24, aime25, ifeval, livebench, "
             "livecodebench, tool_use, memagent",
    )
    parser.add_argument("--tp", type=int, default=4,
                        help="Tensor parallel size (default: 4)")
    parser.add_argument("--n_gpus", type=int, default=0,
                        help="Total GPUs for VERL (0=auto-detect, default: 0)")
    parser.add_argument("--gpu_per_engine", type=int, default=1,
                        help="GPUs per vLLM engine for coding benchmarks (default: 1)")
    parser.add_argument("--output_dir", default="",
                        help="Override results output directory")
    parser.add_argument("--run", type=int, default=0,
                        help="Run number (0=auto-increment)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature for coding benchmarks")
    parser.add_argument("--chat_template", default="",
                        help="Chat template (qwen or cure)")
    parser.add_argument("--memagent_tests", default="hqa",
                        help="MemAgent tests: hqa, ruler, or all (default: hqa)")
    parser.add_argument("--memagent_method", default="openai",
                        help="MemAgent method: openai or recurrent (default: openai)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # ── Validate model path ──
    model_path = args.model.rstrip("/")
    if not Path(model_path).exists():
        print(f"ERROR: Model path does not exist: {model_path}")
        sys.exit(1)

    # ── Auto-detect model properties ──
    print("\n" + "=" * 64)
    print("  Model Auto-Detection")
    print("=" * 64)
    model_info = detect_model_info(model_path)

    # ── Auto-detect GPU count ──
    n_gpus = args.n_gpus if args.n_gpus > 0 else detect_gpu_count()
    print(f"  [gpu] Using {n_gpus} GPUs")

    # ── Resolve benchmarks ──
    benchmarks = resolve_benchmarks(args.benchmarks)
    if not benchmarks:
        print("ERROR: No valid benchmarks specified.")
        sys.exit(1)

    # ── Determine run number ──
    results_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    run_num = args.run if args.run > 0 else determine_run_num(results_dir)
    run_tag = f"run_{run_num:03d}"
    run_dir = results_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Build config ──
    config = EvalConfig(
        model=model_path,
        model_info=model_info,
        benchmarks=benchmarks,
        tp=args.tp,
        n_gpus=n_gpus,
        gpu_per_engine=args.gpu_per_engine,
        output_dir=str(run_dir),
        log_dir=str(run_dir / "logs"),
        run_num=run_num,
        temperature=args.temperature,
        chat_template=args.chat_template,
        memagent_tests=args.memagent_tests,
        memagent_method=args.memagent_method,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    # ── Print header ──
    print("\n" + "=" * 64)
    print("  Unified Evaluation Runner")
    print("=" * 64)
    print(f"  Model:          {model_path}")
    print(f"  Short name:     {model_info.short_name}")
    print(f"  Benchmarks:     {', '.join(benchmarks)}")
    print(f"  Run:            {run_tag} -> {run_dir}")
    print(f"  TP:             {config.tp}")
    print(f"  N GPUs:         {n_gpus}")
    print(f"  GPU/engine:     {config.gpu_per_engine} (coding)")
    print(f"  Chat template:  {config.chat_template or 'default'}")
    print(f"  max_pos_emb:    {model_info.max_position_embeddings}")
    print(f"  Started:        {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)

    # ── Save config for reproducibility ──
    config_record = {
        "model": model_path,
        "model_short": model_info.short_name,
        "benchmarks": benchmarks,
        "tp": config.tp,
        "n_gpus": n_gpus,
        "gpu_per_engine": config.gpu_per_engine,
        "chat_template": config.chat_template,
        "memagent_tests": config.memagent_tests,
        "memagent_method": config.memagent_method,
        "temperature": config.temperature,
        "max_position_embeddings": model_info.max_position_embeddings,
        "has_rope_scaling": model_info.has_rope_scaling,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(run_dir / "eval_config.json", "w") as f:
            json.dump(config_record, f, indent=2)
    except OSError:
        pass

    # ── Run benchmarks ──
    start_time = time.time()
    results = {}

    # Dispatch table: benchmark name -> (runner function, extra args)
    dispatch = {
        "aime24":        lambda: run_aime(config, "aime24"),
        "aime25":        lambda: run_aime(config, "aime25"),
        "aime26":        lambda: run_aime(config, "aime26"),
        "ifeval":        lambda: run_ifeval(config),
        "livebench":     lambda: run_coding(config, "LiveBench"),
        "livecodebench": lambda: run_coding(config, "LiveCodeBench"),
        "tool_use":      lambda: run_tool_use(config),
        "memagent":      lambda: run_memagent(config),
    }

    for bench in benchmarks:
        print(f"\n{'=' * 64}")
        print(f"  [{bench}] Starting...")
        print(f"{'=' * 64}")

        bench_start = time.time()
        runner = dispatch.get(bench)
        if runner is None:
            results[bench] = {"status": "error", "message": f"No runner for {bench}"}
            continue

        try:
            result = runner()
        except Exception as e:
            result = {"status": "error", "message": str(e)}
            print(f"  [ERROR] {bench} raised exception: {e}")

        bench_elapsed = time.time() - bench_start
        result["elapsed_seconds"] = round(bench_elapsed, 1)
        results[bench] = result

        # Collect results into run directory
        if result.get("status") == "success":
            if bench == "livebench":
                collect_coding_result(config, run_dir, "LiveBench")
            elif bench == "livecodebench":
                collect_coding_result(config, run_dir, "LiveCodeBench")
            elif bench == "tool_use":
                collect_tool_use_result(config, run_dir)
            elif bench == "memagent":
                collect_memagent_result(config, run_dir)


    # ── Summary ──
    elapsed = time.time() - start_time
    print_summary(results, elapsed, run_dir)

    # ── Exit code: non-zero if any benchmark failed ──
    failed = [b for b, r in results.items() if r.get("status") != "success"]
    if failed:
        print(f"\n  WARNING: {len(failed)} benchmark(s) failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
