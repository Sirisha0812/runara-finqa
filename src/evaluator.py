"""Evaluation utilities for the FinQA chatbot."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agent import FinQAAgent
from src.data_loader import load_finqa_dataset
from src.logger import get_logger
from src.retriever import FinQARetriever

logger = get_logger(__name__)


PROGRAM_OPERATORS = (
    "add",
    "subtract",
    "multiply",
    "divide",
    "exp",
    "greater",
    "table_max",
    "table_min",
    "table_sum",
    "table_average",
)


class FinQAEvaluator:
    """Runs FinQA evaluation and aggregates answer, retrieval, and reasoning metrics."""

    def __init__(self, output_path: str = "data/eval_results.json") -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent: Optional[FinQAAgent] = None
        self.retriever = FinQARetriever()

    def _init_agent(self) -> None:
        if self.agent is None:
            self.agent = FinQAAgent()

    @staticmethod
    def _normalize_text(text: str) -> str:
        cleaned = text.lower().strip()
        cleaned = re.sub(r"[\$,]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        match = re.search(r"FINAL_ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        return match.group(1).strip() if match else text.strip()

    @staticmethod
    def _parse_numeric_value(text: str) -> Tuple[Optional[float], Optional[str]]:
        normalized = text.lower().replace(",", "").strip()
        multiplier = 1.0
        unit = "absolute"

        if "%" in normalized or "percent" in normalized:
            unit = "percent"
        elif "billion" in normalized or re.search(r"\b\d+(\.\d+)?b\b", normalized):
            multiplier = 1_000_000_000.0
        elif "million" in normalized or re.search(r"\b\d+(\.\d+)?m\b", normalized):
            multiplier = 1_000_000.0
        elif "thousand" in normalized or re.search(r"\b\d+(\.\d+)?k\b", normalized):
            multiplier = 1_000.0

        match = re.search(r"-?\d+(?:\.\d+)?", normalized)
        if not match:
            return None, unit

        try:
            return float(match.group()) * multiplier, unit
        except ValueError:
            return None, unit

    def compare_answers(
        self,
        predicted: str,
        gold: str,
        tolerance: float = 0.01,
    ) -> Dict[str, bool]:
        predicted_clean = self._extract_final_answer(predicted)
        pred_norm = self._normalize_text(predicted_clean)
        gold_norm = self._normalize_text(gold)

        metrics = {
            "exact_match": pred_norm == gold_norm,
            "numeric_match": False,
            "tolerance_match": False,
        }

        pred_num, pred_unit = self._parse_numeric_value(predicted_clean)
        gold_num, gold_unit = self._parse_numeric_value(gold)

        if pred_num is not None and gold_num is not None and pred_unit == gold_unit:
            if abs(pred_num - gold_num) < 1e-6:
                metrics["numeric_match"] = True
                metrics["tolerance_match"] = True
            else:
                denominator = max(abs(gold_num), 1e-8)
                relative_error = abs(pred_num - gold_num) / denominator
                metrics["tolerance_match"] = relative_error <= tolerance

        return metrics

    @staticmethod
    def _program_ops(program: str) -> List[str]:
        return re.findall(r"(add|subtract|multiply|divide|exp|greater|table_max|table_min|table_sum|table_average)", program)

    @staticmethod
    def _expr_ops(expression: Optional[str]) -> List[str]:
        if not expression:
            return []
        mapping = {
            "+": "add",
            "-": "subtract",
            "*": "multiply",
            "/": "divide",
        }
        return [mapping[token] for token in re.findall(r"[\+\-\*/]", expression) if token in mapping]

    def evaluate_example(self, example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        question = example["question"]
        gold_answer = str(example.get("exe_ans") or example.get("answer") or "")
        gold_program = str(example.get("program") or "")

        logger.info("evaluate_example_started", index=idx, question=question[:120])

        start_time = time.time()
        result = self.agent.run(question, example)
        latency_ms = (time.time() - start_time) * 1000

        answer_metrics = self.compare_answers(result["final_answer"], gold_answer)
        retrieved_chunks = result.get("retrieved_chunks", [])
        answer_support_at_k = self.retriever.answer_in_top_k(
            example=example,
            question=question,
            k=min(5, max(len(retrieved_chunks), 1)),
        )

        gold_ops = self._program_ops(gold_program)
        predicted_ops = self._expr_ops(result.get("calculation_expression"))

        calculator_used = bool(result.get("calculation_expression"))
        should_calculate = bool(gold_ops)
        op_overlap = len(set(gold_ops) & set(predicted_ops))
        op_union = len(set(gold_ops) | set(predicted_ops))

        evaluation = {
            "example_idx": idx,
            "question": question,
            "gold_answer": gold_answer,
            "gold_program": gold_program,
            "predicted_answer": result["final_answer"],
            "predicted_calculation_expression": result.get("calculation_expression"),
            "predicted_calculation_result": result.get("calculation_result"),
            "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in retrieved_chunks],
            "retrieval_answer_support_at_5": answer_support_at_k,
            "exact_match": answer_metrics["exact_match"],
            "numeric_match": answer_metrics["numeric_match"],
            "tolerance_match": answer_metrics["tolerance_match"],
            "verification_status": result["verification_status"],
            "verification_confidence": result["verification_confidence"],
            "retry_count": result["retry_count"],
            "calculator_used": calculator_used,
            "gold_requires_calculation": should_calculate,
            "operator_match": set(gold_ops) == set(predicted_ops) if gold_ops or predicted_ops else True,
            "operator_jaccard": round((op_overlap / op_union), 4) if op_union else 1.0,
            "latency_ms": round(latency_ms, 2),
        }

        logger.info(
            "evaluate_example_completed",
            index=idx,
            tolerance_match=evaluation["tolerance_match"],
            retrieval_answer_support_at_5=answer_support_at_k,
            operator_jaccard=evaluation["operator_jaccard"],
        )
        return evaluation

    def evaluate(self, split: str = "validation", num_examples: int = 50) -> Dict[str, Any]:
        logger.info("evaluation_started", split=split, num_examples=num_examples)
        self._init_agent()

        dataset = load_finqa_dataset(split=split)
        examples = [dataset[i] for i in range(min(num_examples, len(dataset)))]

        results = [self.evaluate_example(example, idx) for idx, example in enumerate(examples)]
        metrics = self.calculate_metrics(results)
        operator_distribution = self._operator_distribution(examples)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "split": split,
            "num_examples": len(results),
            "metrics": metrics,
            "operator_distribution": operator_distribution,
            "results": results,
        }

        with open(self.output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        logger.info(
            "evaluation_completed",
            split=split,
            num_examples=len(results),
            output_path=str(self.output_path),
            **metrics,
        )
        return summary

    @staticmethod
    def _operator_distribution(examples: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = Counter()
        for example in examples:
            counts.update(FinQAEvaluator._program_ops(str(example.get("program") or "")))
        return dict(counts)

    @staticmethod
    def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
        total = len(results)
        if total == 0:
            return {}

        def rate(key: str, expected: Any = True) -> float:
            return round(sum(1 for row in results if row.get(key) == expected) / total, 4)

        valid_latencies = [row["latency_ms"] for row in results]
        operator_jaccards = [row["operator_jaccard"] for row in results]

        calculator_tp = sum(
            1
            for row in results
            if row["calculator_used"] and row["gold_requires_calculation"]
        )
        calculator_fp = sum(
            1
            for row in results
            if row["calculator_used"] and not row["gold_requires_calculation"]
        )
        calculator_fn = sum(
            1
            for row in results
            if not row["calculator_used"] and row["gold_requires_calculation"]
        )

        precision = calculator_tp / max(calculator_tp + calculator_fp, 1)
        recall = calculator_tp / max(calculator_tp + calculator_fn, 1)

        verification_pass_rate = rate("verification_status", "PASS")
        verification_fail_rate = rate("verification_status", "FAIL")
        verification_uncertain_rate = rate("verification_status", "UNCERTAIN")

        return {
            "exact_match_rate": rate("exact_match"),
            "numeric_match_rate": rate("numeric_match"),
            "tolerance_match_rate": rate("tolerance_match"),
            "retrieval_answer_support_at_5": rate("retrieval_answer_support_at_5"),
            "verification_pass_rate": verification_pass_rate,
            "verification_fail_rate": verification_fail_rate,
            "verification_uncertain_rate": verification_uncertain_rate,
            "calculator_usage_rate": rate("calculator_used"),
            "calculator_precision": round(precision, 4),
            "calculator_recall": round(recall, 4),
            "operator_match_rate": rate("operator_match"),
            "avg_operator_jaccard": round(sum(operator_jaccards) / total, 4),
            "avg_latency_ms": round(sum(valid_latencies) / total, 2),
            "avg_retry_count": round(sum(row["retry_count"] for row in results) / total, 3),
        }

    @staticmethod
    def print_summary(summary: Dict[str, Any]) -> None:
        metrics = summary["metrics"]
        print("\n" + "=" * 80)
        print("FINQA EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Split: {summary['split']}")
        print(f"Examples: {summary['num_examples']}")
        print(f"Timestamp: {summary['timestamp']}")
        print("\nAnswer Metrics")
        print(f"  Exact match rate:        {metrics['exact_match_rate']:.1%}")
        print(f"  Numeric match rate:      {metrics['numeric_match_rate']:.1%}")
        print(f"  Tolerance match rate:    {metrics['tolerance_match_rate']:.1%}")
        print("\nRetrieval Metrics")
        print(f"  Answer support@5:        {metrics['retrieval_answer_support_at_5']:.1%}")
        print("\nReasoning Metrics")
        print(f"  Calculator precision:    {metrics['calculator_precision']:.1%}")
        print(f"  Calculator recall:       {metrics['calculator_recall']:.1%}")
        print(f"  Operator match rate:     {metrics['operator_match_rate']:.1%}")
        print(f"  Avg operator jaccard:    {metrics['avg_operator_jaccard']:.3f}")
        print("\nWorkflow Metrics")
        print(f"  Verification pass rate:  {metrics['verification_pass_rate']:.1%}")
        print(f"  Verification fail rate:  {metrics['verification_fail_rate']:.1%}")
        print(f"  Verification uncertain:  {metrics['verification_uncertain_rate']:.1%}")
        print(f"  Avg retry count:         {metrics['avg_retry_count']:.2f}")
        print(f"  Avg latency:             {metrics['avg_latency_ms']:.0f} ms")
        print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the FinQA chatbot")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--output", type=str, default="data/eval_results.json")
    args = parser.parse_args()

    evaluator = FinQAEvaluator(output_path=args.output)
    summary = evaluator.evaluate(split=args.split, num_examples=args.num_examples)
    evaluator.print_summary(summary)
