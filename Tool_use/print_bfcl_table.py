#!/usr/bin/env python3
"""Print BFCL score summary in the standard Live (Wgt) / Non-Live (Unwgt) table format."""

import argparse
import json
import sys
from pathlib import Path

# Live categories (AST only, irrelevance/relevance excluded)
LIVE_CATS = [
    ("Simple", "live_simple", 258),
    ("Multiple", "live_multiple", 1053),
    ("Parallel", "live_parallel", 16),
    ("ParMul", "live_parallel_multiple", 24),
]

# Non-Live categories (irrelevance excluded)
NONLIVE_CATS = [
    ("Py Simple", "simple", None),
    ("Java Simple", "java", None),
    ("JS Simple", "javascript", None),
    ("Multiple", "multiple", None),
    ("Parallel", "parallel", None),
    ("ParMul", "parallel_multiple", None),
]


def parse_bfcl_scores(score_dir: Path) -> dict:
    scores = {}
    for f in sorted(score_dir.iterdir()):
        if not f.name.endswith("_score.json"):
            continue
        try:
            first_line = json.loads(f.read_text().split("\n", 1)[0])
            if "accuracy" in first_line:
                cat = f.name.replace("BFCL_v3_", "").replace("_score.json", "")
                scores[cat] = first_line["accuracy"]
        except (json.JSONDecodeError, OSError):
            pass
    return scores


def print_table(model_short: str, scores: dict):
    def fmt(key):
        v = scores.get(key)
        return f"{v*100:.1f}%" if v is not None else "N/A"

    # Live row (weighted average)
    live_header = "Model\t" + "\t".join(
        f"{name} ({n})" for name, _, n in LIVE_CATS
    ) + "\tLive (Wgt)"

    live_vals = [scores.get(key, 0) for _, key, _ in LIVE_CATS]
    live_weights = [n for _, _, n in LIVE_CATS]
    live_wgt = (sum(v * w for v, w in zip(live_vals, live_weights))
                / sum(live_weights)) if sum(live_weights) > 0 else 0

    live_row = model_short + "\t" + "\t".join(fmt(key) for _, key, _ in LIVE_CATS)
    live_row += f"\t{live_wgt*100:.1f}%"

    # Non-Live row (unweighted average)
    nonlive_header = "Model\t" + "\t".join(
        name for name, _, _ in NONLIVE_CATS
    ) + "\tNon-Live (Unwgt)"

    nonlive_vals = [scores.get(key, 0) for _, key, _ in NONLIVE_CATS]
    nonlive_avg = sum(nonlive_vals) / len(nonlive_vals) if nonlive_vals else 0

    nonlive_row = model_short + "\t" + "\t".join(fmt(key) for _, key, _ in NONLIVE_CATS)
    nonlive_row += f"\t{nonlive_avg*100:.1f}%"

    print()
    print("[Tool Use - Live]")
    print(live_header)
    print(live_row)
    print()
    print("[Tool Use - Non-Live]")
    print(nonlive_header)
    print(nonlive_row)
    print()


def main():
    parser = argparse.ArgumentParser(description="Print BFCL score table")
    parser.add_argument("model_name", help="Model short name")
    parser.add_argument("--score-dir", type=Path, required=True,
                        help="Path to score directory for this model")
    args = parser.parse_args()

    if not args.score_dir.exists():
        print(f"Score directory not found: {args.score_dir}", file=sys.stderr)
        sys.exit(1)

    scores = parse_bfcl_scores(args.score_dir)
    if not scores:
        print(f"No score files found in {args.score_dir}", file=sys.stderr)
        sys.exit(1)

    print_table(args.model_name, scores)


if __name__ == "__main__":
    main()
