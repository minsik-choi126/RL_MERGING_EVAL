#!/usr/bin/env python3
"""
BFCL 결과 요약 스크립트
CSV 파일을 파싱하여 요청한 형태로 출력합니다.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional


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


def summarize_results(model_name: str, score_dir: Path) -> None:
    """결과를 요약하여 출력"""

    # CSV 파일 읽기
    non_live_row = read_csv_row(score_dir / 'data_non_live.csv', model_name)
    live_row = read_csv_row(score_dir / 'data_live.csv', model_name)
    overall_row = read_csv_row(score_dir / 'data_overall.csv', model_name)

    if not non_live_row and not live_row and not overall_row:
        print(f"❌ Error: No results found for model '{model_name}'", file=sys.stderr)
        sys.exit(1)

    print("=" * 120)
    print(f"  BFCL Tool-Use Evaluation Results: {model_name}")
    print("=" * 120)
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # Non-Live Results
    # ──────────────────────────────────────────────────────────────────────────
    if non_live_row:
        print("📊 Non-Live Results")
        print("-" * 120)

        # 열 이름 출력
        headers = ['irrelevance', 'java', 'js', 'multi', 'para', 'para_multi', 'simple', 'Avg']
        print(f"  {'Category':<15} " + " ".join(f"{h:>10}" for h in headers))
        print(f"  {'-'*15} " + " ".join(f"{'-'*10}" for _ in headers))

        # 값 추출
        values = {
            'irrelevance': parse_percentage(non_live_row.get('Irrelevance Detection', 'N/A')),
            'java': parse_percentage(non_live_row.get('Java Simple AST', 'N/A')),
            'js': parse_percentage(non_live_row.get('JavaScript Simple AST', 'N/A')),
            'multi': parse_percentage(non_live_row.get('Multiple AST', 'N/A')),
            'para': parse_percentage(non_live_row.get('Parallel AST', 'N/A')),
            'para_multi': parse_percentage(non_live_row.get('Parallel Multiple AST', 'N/A')),
            'simple': parse_percentage(non_live_row.get('Simple AST', 'N/A')),
        }

        # 평균 계산 (N/A 제외)
        valid_values = [v for v in values.values() if v is not None]
        avg = sum(valid_values) / len(valid_values) if valid_values else None

        # 값 출력
        value_strs = []
        for key in headers[:-1]:  # Avg 제외
            val = values[key]
            value_strs.append(f"{val:>9.2f}%" if val is not None else f"{'N/A':>10}")
        value_strs.append(f"{avg:>9.2f}%" if avg is not None else f"{'N/A':>10}")

        print(f"  {'Scores':<15} " + " ".join(value_strs))
        print()

    # ──────────────────────────────────────────────────────────────────────────
    # Live Results
    # ──────────────────────────────────────────────────────────────────────────
    if live_row:
        print("📊 Live Results")
        print("-" * 120)

        # 열 이름 출력
        headers = ['live_irrel', 'live_multi', 'live_para', 'live_pm', 'live_rel', 'live_simple', 'Avg']
        print(f"  {'Category':<15} " + " ".join(f"{h:>10}" for h in headers))
        print(f"  {'-'*15} " + " ".join(f"{'-'*10}" for _ in headers))

        # 값 추출
        values = {
            'live_irrel': parse_percentage(live_row.get('Irrelevance Detection', 'N/A')),
            'live_multi': parse_percentage(live_row.get('Python Multiple AST', 'N/A')),
            'live_para': parse_percentage(live_row.get('Python Parallel AST', 'N/A')),
            'live_pm': parse_percentage(live_row.get('Python Parallel Multiple AST', 'N/A')),
            'live_rel': parse_percentage(live_row.get('Relevance Detection', 'N/A')),
            'live_simple': parse_percentage(live_row.get('Python Simple AST', 'N/A')),
        }

        # 평균 계산 (N/A 제외)
        valid_values = [v for v in values.values() if v is not None]
        avg = sum(valid_values) / len(valid_values) if valid_values else None

        # 값 출력
        value_strs = []
        for key in headers[:-1]:  # Avg 제외
            val = values[key]
            value_strs.append(f"{val:>9.2f}%" if val is not None else f"{'N/A':>10}")
        value_strs.append(f"{avg:>9.2f}%" if avg is not None else f"{'N/A':>10}")

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
        description='Summarize BFCL evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 summarize_results.py whitened-k512
  python3 summarize_results.py RAM-plus
  python3 summarize_results.py Qwen3-1.7B-base
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
