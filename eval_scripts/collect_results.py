#!/usr/bin/env python3
"""
Collect VERL evaluation results from multiple runs and generate an Excel summary.

Reads metrics from:
  1. VERL FileLogger JSONL files (metrics.jsonl) — preferred
  2. Console log files (eval.log) — fallback (parses "step:N - key:val" lines)

Output:
  {output_root}/eval_summary.xlsx with per-run and averaged results.

Usage:
  python3 collect_results.py --output-root /path/to/eval_outputs --n-repeats 3
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Key metrics to extract (in display order).
# Each entry: (display_name, list_of_possible_metric_key_patterns)
# Patterns are matched as substrings against the full metric key.
METRIC_PATTERNS = [
    ("acc/mean@8", ["acc/mean@8"]),
    ("acc/best@8", ["acc/best@8/mean"]),
    ("acc/maj@8", ["acc/maj@8/mean"]),
    ("acc/mean@1", ["acc/mean@1"]),
    ("reward/mean@8", ["reward/mean@8"]),
    ("instruction_pass_ratio/best@8", ["instruction_pass_ratio/best@8/mean"]),
    ("instruction_pass_ratio/mean@8", ["instruction_pass_ratio/mean@8"]),
]


def parse_jsonl(filepath: str) -> dict:
    """Parse VERL FileLogger JSONL output."""
    metrics = {}
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if "data" in entry and isinstance(entry["data"], dict):
                    metrics.update(entry["data"])
    except Exception:
        pass
    return metrics


def parse_console_log(filepath: str) -> dict:
    """Parse console log for 'step:N - key:val - ...' lines."""
    metrics = {}
    step_pattern = re.compile(r"step:\d+\s*-\s*(.+)$")
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                m = step_pattern.search(line)
                if not m:
                    continue
                parts = m.group(1).split(" - ")
                for part in parts:
                    if ":" in part:
                        key, _, val = part.partition(":")
                        key = key.strip()
                        val = val.strip()
                        try:
                            metrics[key] = float(val)
                        except ValueError:
                            pass
    except Exception:
        pass
    return metrics


def extract_metrics(run_dir: str) -> dict:
    """Extract metrics from a single run directory."""
    jsonl_path = os.path.join(run_dir, "metrics.jsonl")
    log_path = os.path.join(run_dir, "eval.log")

    metrics = {}
    if os.path.isfile(jsonl_path) and os.path.getsize(jsonl_path) > 0:
        metrics = parse_jsonl(jsonl_path)
    if not metrics and os.path.isfile(log_path):
        metrics = parse_console_log(log_path)
    return metrics


def match_metric(full_key: str, patterns: list[str]) -> bool:
    """Check if a full metric key matches any of the patterns."""
    for pat in patterns:
        if pat in full_key:
            return True
    return False


def find_metric_value(all_metrics: dict, patterns: list[str]) -> float | None:
    """Find the first matching metric value from patterns."""
    for key, val in all_metrics.items():
        if match_metric(key, patterns):
            return val
    return None


def discover_runs(output_root: str) -> list[dict]:
    """Discover all run directories and extract model/benchmark/run info."""
    runs = []
    root = Path(output_root)
    if not root.exists():
        return runs

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        eval_dir = model_dir / "evaluation_output"
        if not eval_dir.exists():
            continue
        model_name = model_dir.name

        for bench_dir in sorted(eval_dir.iterdir()):
            if not bench_dir.is_dir():
                continue
            benchmark = bench_dir.name

            for run_dir in sorted(bench_dir.iterdir()):
                if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                    continue
                run_idx = run_dir.name.replace("run_", "")
                try:
                    run_idx = int(run_idx)
                except ValueError:
                    continue

                metrics = extract_metrics(str(run_dir))
                if metrics:
                    runs.append({
                        "model": model_name,
                        "benchmark": benchmark,
                        "run": run_idx,
                        "raw_metrics": metrics,
                    })
    return runs


def build_dataframe(runs: list[dict]) -> pd.DataFrame:
    """Build a flat DataFrame from discovered runs."""
    rows = []
    for run_info in runs:
        row = {
            "Model": run_info["model"],
            "Benchmark": run_info["benchmark"],
            "Run": run_info["run"],
        }
        for display_name, patterns in METRIC_PATTERNS:
            val = find_metric_value(run_info["raw_metrics"], patterns)
            if val is not None:
                row[display_name] = val
        rows.append(row)
    return pd.DataFrame(rows)


def generate_excel(df: pd.DataFrame, output_path: str):
    """Generate Excel with per-run sheet and average sheet."""
    if df.empty:
        print("No results found. Excel not generated.")
        return

    # Metric columns (exclude Model, Benchmark, Run)
    metric_cols = [c for c in df.columns if c not in ("Model", "Benchmark", "Run")]

    # Per-run data (sorted)
    df_sorted = df.sort_values(["Model", "Benchmark", "Run"]).reset_index(drop=True)

    # Average across runs
    group_cols = ["Model", "Benchmark"]
    df_avg = df_sorted.groupby(group_cols, sort=True)[metric_cols].mean().reset_index()
    df_std = df_sorted.groupby(group_cols, sort=True)[metric_cols].std().reset_index()

    # Rename std columns
    std_rename = {c: f"{c} (std)" for c in metric_cols}
    df_std = df_std.rename(columns=std_rename)

    # Merge avg and std
    df_summary = pd.merge(df_avg, df_std, on=group_cols)

    # Reorder columns: for each metric, put avg then std next to each other
    ordered_cols = list(group_cols)
    for c in metric_cols:
        ordered_cols.append(c)
        std_col = f"{c} (std)"
        if std_col in df_summary.columns:
            ordered_cols.append(std_col)
    df_summary = df_summary[[c for c in ordered_cols if c in df_summary.columns]]

    # Also create a pivot-style view: models as rows, benchmarks × metrics as columns
    pivot_rows = []
    for model in df_avg["Model"].unique():
        row = {"Model": model}
        for _, bench_row in df_avg[df_avg["Model"] == model].iterrows():
            bench = bench_row["Benchmark"]
            for mc in metric_cols:
                if pd.notna(bench_row.get(mc)):
                    row[f"{bench}/{mc}"] = bench_row[mc]
            # Add std
            std_match = df_std[
                (df_std["Model"] == model) & (df_std["Benchmark"] == bench)
            ]
            if not std_match.empty:
                for mc in metric_cols:
                    std_col = f"{mc} (std)"
                    if std_col in std_match.columns and pd.notna(std_match.iloc[0].get(std_col)):
                        row[f"{bench}/{mc} (std)"] = std_match.iloc[0][std_col]
        pivot_rows.append(row)
    df_pivot = pd.DataFrame(pivot_rows)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_sorted.to_excel(writer, sheet_name="Per-Run Results", index=False)
        df_summary.to_excel(writer, sheet_name="Average (per benchmark)", index=False)
        df_pivot.to_excel(writer, sheet_name="Pivot (model × bench)", index=False)

        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col_cells in ws.columns:
                max_len = 0
                col_letter = col_cells[0].column_letter
                for cell in col_cells:
                    try:
                        cell_len = len(str(cell.value)) if cell.value else 0
                        max_len = max(max_len, cell_len)
                    except Exception:
                        pass
                ws.column_dimensions[col_letter].width = min(max_len + 3, 40)

    print(f"Excel saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect VERL eval results into Excel")
    parser.add_argument("--output-root", required=True, help="Root directory of eval outputs")
    parser.add_argument("--n-repeats", type=int, default=3, help="Number of repeats (for info only)")
    parser.add_argument("--excel-path", default=None, help="Output Excel path (default: {output-root}/eval_summary.xlsx)")
    args = parser.parse_args()

    output_root = args.output_root
    excel_path = args.excel_path or os.path.join(output_root, "eval_summary.xlsx")

    print(f"Scanning: {output_root}")
    runs = discover_runs(output_root)
    print(f"Found {len(runs)} completed run(s)")

    if not runs:
        print("No results to collect. Make sure evaluations have completed.")
        return

    # Show discovered runs
    for r in runs:
        metric_count = len([v for v in r["raw_metrics"].values() if isinstance(v, (int, float))])
        print(f"  {r['model']} / {r['benchmark']} / run_{r['run']} — {metric_count} metrics")

    df = build_dataframe(runs)
    print(f"\nDataFrame shape: {df.shape}")
    print(df.to_string(index=False))

    generate_excel(df, excel_path)


if __name__ == "__main__":
    main()
