#!/usr/bin/env python3
"""
BFCL 결과 요약 스크립트 (Weighted Averaging 방식)
BFCL 리더보드와 동일한 weighted averaging 방식으로 점수 계산
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Sample Counts (from BFCL_v3 dataset)
# ─────────────────────────────────────────────────────────────────────────────

# Live categories sample counts (AST only, irrelevance/relevance excluded)
LIVE_WEIGHTS = {
    'live_simple': 258,
    'live_multiple': 1053,
    'live_parallel': 16,
    'live_parallel_multiple': 24,
}

# Non-Live categories sample counts (irrelevance excluded)
NON_LIVE_WEIGHTS = {
    'simple': 400,  # Python Simple AST
    'java': 100,
    'javascript': 50,
    'multiple': 200,
    'parallel': 200,
    'parallel_multiple': 200,
}


def parse_percentage(value: str) -> Optional[float]:
    """백분율 문자열을 float로 변환 (예: '75.39%' -> 75.39)"""
    if not value or value == 'N/A':
        return None
    return float(value.rstrip('%'))


def read_csv_row(csv_path: Path, model_name: str) -> Optional[Dict[str, str]]:
    """CSV 파일에서 특정 모델의 행을 찾아 반환"""
    if not csv_path.exists():
        return None

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Model', '').startswith(model_name):
                return row
    return None


def calculate_weighted_average(
    scores: Dict[str, Optional[float]],
    weights: Dict[str, int]
) -> Optional[float]:
    """
    가중 평균 계산

    Args:
        scores: {category: score} 딕셔너리
        weights: {category: sample_count} 딕셔너리

    Returns:
        weighted average 또는 None (점수가 없는 경우)
    """
    total_weight = 0
    weighted_sum = 0.0

    for category, weight in weights.items():
        score = scores.get(category)
        if score is not None:
            weighted_sum += score * weight
            total_weight += weight

    if total_weight == 0:
        return None

    return weighted_sum / total_weight


def summarize_results(model_name: str, score_dir: Path) -> None:
    """결과를 요약하여 출력 (Weighted Averaging 방식)"""

    # CSV 파일 읽기
    non_live_row = read_csv_row(score_dir / 'data_non_live.csv', model_name)
    live_row = read_csv_row(score_dir / 'data_live.csv', model_name)
    overall_row = read_csv_row(score_dir / 'data_overall.csv', model_name)

    if not non_live_row and not live_row and not overall_row:
        print(f"❌ Error: No results found for model '{model_name}'", file=sys.stderr)
        sys.exit(1)

    print("=" * 120)
    print(f"  BFCL Tool-Use Evaluation Results: {model_name}")
    print(f"  (Weighted Averaging - BFCL Official Method)")
    print("=" * 120)
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # Non-Live Results (Unweighted Average per BFCL)
    # ──────────────────────────────────────────────────────────────────────────
    if non_live_row:
        print("📊 Non-Live Results (Unweighted Average)")
        print("-" * 120)

        # 열 이름과 샘플 수 (irrelevance 제외)
        headers = ['py_simple', 'java', 'js', 'multi', 'para', 'para_multi', 'Avg']
        sample_counts = [
            NON_LIVE_WEIGHTS['simple'],
            NON_LIVE_WEIGHTS['java'],
            NON_LIVE_WEIGHTS['javascript'],
            NON_LIVE_WEIGHTS['multiple'],
            NON_LIVE_WEIGHTS['parallel'],
            NON_LIVE_WEIGHTS['parallel_multiple'],
            sum(NON_LIVE_WEIGHTS.values())
        ]

        # 헤더 출력 (카테고리명)
        print(f"  {'Category':<15} " + " ".join(f"{h:>10}" for h in headers))
        # 샘플 수 출력
        print(f"  {'(samples)':<15} " + " ".join(f"({n:>8})" for n in sample_counts))
        print(f"  {'-'*15} " + " ".join(f"{'-'*10}" for _ in headers))

        # 값 추출 (irrelevance 제외)
        scores = {
            'simple': parse_percentage(non_live_row.get('Python Simple AST', 'N/A')),
            'java': parse_percentage(non_live_row.get('Java Simple AST', 'N/A')),
            'javascript': parse_percentage(non_live_row.get('JavaScript Simple AST', 'N/A')),
            'multiple': parse_percentage(non_live_row.get('Multiple AST', 'N/A')),
            'parallel': parse_percentage(non_live_row.get('Parallel AST', 'N/A')),
            'parallel_multiple': parse_percentage(non_live_row.get('Parallel Multiple AST', 'N/A')),
        }

        # Unweighted 평균 계산 (irrelevance 제외)
        valid_values = [v for v in scores.values() if v is not None]
        avg = sum(valid_values) / len(valid_values) if valid_values else None

        # 값 출력
        value_strs = []
        for key in ['simple', 'java', 'javascript', 'multiple', 'parallel', 'parallel_multiple']:
            val = scores[key]
            value_strs.append(f"{val:>9.2f}%" if val is not None else f"{'N/A':>10}")
        value_strs.append(f"{avg:>9.2f}%" if avg is not None else f"{'N/A':>10}")

        print(f"  {'Scores':<15} " + " ".join(value_strs))
        print()

    # ──────────────────────────────────────────────────────────────────────────
    # Live Results (Weighted Average)
    # ──────────────────────────────────────────────────────────────────────────
    if live_row:
        print("📊 Live Results (Weighted Average)")
        print("-" * 120)

        # 열 이름과 샘플 수 (AST only, irrelevance/relevance 제외)
        headers = ['live_simple', 'live_multi', 'live_para', 'live_pm', 'Live(Wgt)']
        sample_counts = [
            LIVE_WEIGHTS['live_simple'],
            LIVE_WEIGHTS['live_multiple'],
            LIVE_WEIGHTS['live_parallel'],
            LIVE_WEIGHTS['live_parallel_multiple'],
            sum(LIVE_WEIGHTS.values())
        ]

        # 헤더 출력 (카테고리명)
        print(f"  {'Category':<15} " + " ".join(f"{h:>10}" for h in headers))
        # 샘플 수 출력
        print(f"  {'(samples)':<15} " + " ".join(f"({n:>8})" for n in sample_counts))
        print(f"  {'-'*15} " + " ".join(f"{'-'*10}" for _ in headers))

        # 값 추출 (AST only)
        scores = {
            'live_simple': parse_percentage(live_row.get('Python Simple AST', 'N/A')),
            'live_multiple': parse_percentage(live_row.get('Python Multiple AST', 'N/A')),
            'live_parallel': parse_percentage(live_row.get('Python Parallel AST', 'N/A')),
            'live_parallel_multiple': parse_percentage(live_row.get('Python Parallel Multiple AST', 'N/A')),
        }

        # Weighted 평균 계산 (AST 4개만)
        weighted_avg = calculate_weighted_average(scores, LIVE_WEIGHTS)

        # 값 출력
        value_strs = []
        for key in ['live_simple', 'live_multiple', 'live_parallel', 'live_parallel_multiple']:
            val = scores[key]
            value_strs.append(f"{val:>9.2f}%" if val is not None else f"{'N/A':>10}")
        value_strs.append(f"{weighted_avg:>9.2f}%" if weighted_avg is not None else f"{'N/A':>10}")

        print(f"  {'Scores':<15} " + " ".join(value_strs))
        print()

    # ──────────────────────────────────────────────────────────────────────────
    # Overall Summary
    # ──────────────────────────────────────────────────────────────────────────
    if overall_row:
        print("📊 Overall Summary")
        print("-" * 120)

        overall_acc = overall_row.get('Overall Acc', 'N/A')
        non_live_ast = overall_row.get('Non-Live AST Acc', 'N/A')
        live_acc = overall_row.get('Live Acc', 'N/A')

        print(f"  {'Overall Accuracy:':<30} {overall_acc:>10}")
        print(f"  {'Non-Live AST Accuracy:':<30} {non_live_ast:>10}")
        print(f"  {'Live Accuracy:':<30} {live_acc:>10}")
        print()

        # 샘플 수 정보
        total_live = sum(LIVE_WEIGHTS.values())
        total_non_live = sum(NON_LIVE_WEIGHTS.values())
        print(f"  {'Total Live Samples:':<30} {total_live:>10,}")
        print(f"  {'Total Non-Live Samples:':<30} {total_non_live:>10,}")
        print()

    # ──────────────────────────────────────────────────────────────────────────
    # File Locations
    # ──────────────────────────────────────────────────────────────────────────
    print("📁 Result Files")
    print("-" * 120)
    print(f"  Score directory: {score_dir}")
    print(f"  - data_non_live.csv")
    print(f"  - data_live.csv")
    print(f"  - data_overall.csv")
    print()

    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(
        description='Summarize BFCL evaluation results with weighted averaging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 summarize_results_weighted.py whitened-k512
  python3 summarize_results_weighted.py RAM-plus
  python3 summarize_results_weighted.py Qwen3-1.7B-base
        """
    )
    parser.add_argument('model_name', help='Model name to summarize')
    parser.add_argument(
        '--score-dir',
        type=Path,
        help='Path to score directory (default: ./berkeley-function-call-leaderboard/score)'
    )

    args = parser.parse_args()

    # score_dir 결정
    if args.score_dir:
        score_dir = args.score_dir
    else:
        script_dir = Path(__file__).parent
        score_dir = script_dir / 'berkeley-function-call-leaderboard' / 'score'

    if not score_dir.exists():
        print(f"❌ Error: Score directory not found: {score_dir}", file=sys.stderr)
        sys.exit(1)

    summarize_results(args.model_name, score_dir)


if __name__ == '__main__':
    main()
