#!/usr/bin/env python3
"""
Collect and analyze IFEval results from 3-run evaluation.

Usage:
    python collect_ifeval_results.py                        # Use default path
    python collect_ifeval_results.py --results_dir <path>  # Custom path
    python collect_ifeval_results.py --export_excel        # Export to Excel

Output:
    - Console: Formatted table with averages and std dev
    - CSV: summary_detailed.csv
    - Excel: summary.xlsx (if --export_excel)
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def find_ifeval_scores(output_dir: Path) -> Tuple[float, float]:
    """Extract instruction_pass and accuracy from VERL log."""

    log_file = output_dir / "ifeval_eval.log"
    if not log_file.exists():
        return None, None

    log_text = log_file.read_text()

    # Search for metrics in log
    # VERL typically logs: {"instruction_pass": 0.7234, "accuracy": 0.8123, ...}
    inst_pass_match = re.search(r'"instruction_pass":\s*([0-9.]+)', log_text)
    accuracy_match = re.search(r'"accuracy":\s*([0-9.]+)', log_text)

    inst_pass = float(inst_pass_match.group(1)) if inst_pass_match else None
    accuracy = float(accuracy_match.group(1)) if accuracy_match else None

    return inst_pass, accuracy


def collect_results(results_base: Path) -> Dict[str, List[Dict]]:
    """Collect all results from directory structure."""

    results = defaultdict(list)

    if not results_base.exists():
        print(f"ERROR: Results directory not found: {results_base}")
        return results

    # Iterate through model directories
    for model_dir in sorted(results_base.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "summary.txt":
            continue

        model_name = model_dir.name

        # Iterate through run directories
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue

            run_num = int(run_dir.name.split("_")[1])

            inst_pass, accuracy = find_ifeval_scores(run_dir)

            if inst_pass is not None and accuracy is not None:
                results[model_name].append({
                    'run': run_num,
                    'instruction_pass': inst_pass,
                    'accuracy': accuracy
                })

    return results


def compute_statistics(results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Compute mean and std for each model."""

    stats_data = []

    model_order = [
        "task_arithmetic",
        "TIES",
        "TSV",
        "whitened_k128",
        "whitened_k256",
        "whitened_k512",
        "code_expert"
    ]

    for model in model_order:
        if model not in results or len(results[model]) == 0:
            stats_data.append({
                'Model': model,
                'Instruction Pass (mean)': None,
                'Instruction Pass (std)': None,
                'Accuracy (mean)': None,
                'Accuracy (std)': None,
                'Runs': 0
            })
            continue

        inst_passes = [r['instruction_pass'] for r in results[model]]
        accuracies = [r['accuracy'] for r in results[model]]

        import statistics

        stats_data.append({
            'Model': model,
            'Instruction Pass (mean)': statistics.mean(inst_passes),
            'Instruction Pass (std)': statistics.stdev(inst_passes) if len(inst_passes) > 1 else 0.0,
            'Accuracy (mean)': statistics.mean(accuracies),
            'Accuracy (std)': statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            'Runs': len(inst_passes)
        })

    return pd.DataFrame(stats_data)


def export_to_csv(results: Dict[str, List[Dict]], stats_df: pd.DataFrame, output_path: Path):
    """Export detailed results to CSV."""

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['Model', 'Run', 'Instruction Pass', 'Accuracy'])

        # Individual run data
        for model in sorted(results.keys()):
            for run_data in sorted(results[model], key=lambda x: x['run']):
                writer.writerow([
                    model,
                    run_data['run'],
                    f"{run_data['instruction_pass']:.4f}",
                    f"{run_data['accuracy']:.4f}"
                ])

        # Blank line
        writer.writerow([])

        # Statistics
        writer.writerow(['=== Statistics (Mean ± Std) ==='])
        writer.writerow(['Model', 'Instruction Pass', 'Accuracy', 'Runs'])

        for _, row in stats_df.iterrows():
            if row['Runs'] > 0:
                writer.writerow([
                    row['Model'],
                    f"{row['Instruction Pass (mean)']:.4f} ± {row['Instruction Pass (std)']:.4f}",
                    f"{row['Accuracy (mean)']:.4f} ± {row['Accuracy (std)']:.4f}",
                    int(row['Runs'])
                ])
            else:
                writer.writerow([row['Model'], 'N/A', 'N/A', 0])

    print(f"CSV exported to: {output_path}")


def export_to_excel(stats_df: pd.DataFrame, results: Dict[str, List[Dict]], output_path: Path):
    """Export results to Excel with multiple sheets."""

    if not PANDAS_AVAILABLE:
        print("ERROR: pandas not installed, cannot export to Excel")
        print("Install with: pip install pandas openpyxl")
        return

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Summary statistics
        stats_df.to_excel(writer, sheet_name='Summary', index=False)

        # Sheet 2: Individual runs
        individual_data = []
        for model in sorted(results.keys()):
            for run_data in sorted(results[model], key=lambda x: x['run']):
                individual_data.append({
                    'Model': model,
                    'Run': run_data['run'],
                    'Instruction Pass': run_data['instruction_pass'],
                    'Accuracy': run_data['accuracy']
                })

        individual_df = pd.DataFrame(individual_data)
        individual_df.to_excel(writer, sheet_name='Individual Runs', index=False)

    print(f"Excel exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect IFEval results from 3-run evaluation"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Base directory containing evaluation results "
             "(e.g., ./results/ifeval_multirun)"
    )
    parser.add_argument(
        "--export_excel", action="store_true",
        help="Export results to Excel (requires pandas)"
    )
    args = parser.parse_args()

    results_base = Path(args.results_dir)

    print("=" * 80)
    print("  IFEval Results Collection")
    print("=" * 80)
    print(f"Results directory: {results_base}")
    print()

    # Collect results
    results = collect_results(results_base)

    if not results:
        print("ERROR: No results found!")
        return 1

    # Compute statistics
    stats_df = compute_statistics(results)

    # Display results
    print("=" * 80)
    print("  Summary (Mean ± Std across 3 runs)")
    print("=" * 80)
    print()
    print(stats_df.to_string(index=False))
    print()

    # Export CSV
    csv_path = results_base / "summary_detailed.csv"
    export_to_csv(results, stats_df, csv_path)

    # Export Excel if requested
    if args.export_excel:
        excel_path = results_base / "summary.xlsx"
        export_to_excel(stats_df, results, excel_path)

    print()
    print("=" * 80)
    print("  Collection complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
