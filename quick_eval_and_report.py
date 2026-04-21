#!/usr/bin/env python3
"""Run evaluation and emit a concise markdown summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluator import FinQAEvaluator


def build_report(summary: dict[str, object]) -> str:
    metrics = summary["metrics"]
    operator_distribution = summary["operator_distribution"]

    top_ops = ", ".join(
        f"{name}: {count}"
        for name, count in list(operator_distribution.items())[:5]
    )

    return f"""# FinQA Evaluation Summary

Split: {summary['split']}
Examples: {summary['num_examples']}
Timestamp: {summary['timestamp']}

## Answer Quality

- Exact match rate: {metrics['exact_match_rate']:.1%}
- Numeric match rate: {metrics['numeric_match_rate']:.1%}
- Tolerance match rate: {metrics['tolerance_match_rate']:.1%}

## Retrieval

- Answer support@5: {metrics['retrieval_answer_support_at_5']:.1%}

## Numerical Reasoning

- Calculator precision: {metrics['calculator_precision']:.1%}
- Calculator recall: {metrics['calculator_recall']:.1%}
- Operator match rate: {metrics['operator_match_rate']:.1%}
- Average operator Jaccard: {metrics['avg_operator_jaccard']:.3f}

## Workflow

- Verification pass rate: {metrics['verification_pass_rate']:.1%}
- Verification fail rate: {metrics['verification_fail_rate']:.1%}
- Verification uncertain rate: {metrics['verification_uncertain_rate']:.1%}
- Average latency: {metrics['avg_latency_ms']:.0f} ms
- Average retry count: {metrics['avg_retry_count']:.2f}

## Operator Mix

{top_ops}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinQA evaluation and build a markdown summary")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"eval_results_{args.split}_{args.num_examples}.json"
    md_path = output_dir / f"eval_summary_{args.split}_{args.num_examples}.md"

    evaluator = FinQAEvaluator(output_path=str(json_path))
    summary = evaluator.evaluate(split=args.split, num_examples=args.num_examples)

    md_path.write_text(build_report(summary), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
