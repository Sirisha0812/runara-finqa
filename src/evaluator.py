"""
FinQA Agent Evaluator

Evaluates agent performance on validation set examples.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.agent import FinQAAgent
from src.data_loader import load_finqa_dataset
from src.logger import get_logger

logger = get_logger(__name__)


class FinQAEvaluator:
    """Evaluator for FinQA agent performance."""

    def __init__(self, output_path: str = "data/eval_results.json"):
        """
        Initialize evaluator.

        Args:
            output_path: Path to save evaluation results
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent = FinQAAgent()

    def extract_number(self, text: str) -> Optional[float]:
        """
        Extract numeric value from text.

        Handles formats like: "45.0", "$45M", "45%", "45.0 million", etc.

        Args:
            text: Text containing number

        Returns:
            Extracted number or None
        """
        if not text:
            return None

        # Remove common text patterns
        text = text.lower().strip()
        text = re.sub(r'\$|,|%', '', text)

        # Try to find a number
        # Match patterns like: 45.0, -45.0, 45, .5, etc.
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None

        return None

    def compare_answers(
        self,
        predicted: str,
        gold: str,
        tolerance: float = 0.01
    ) -> Dict[str, bool]:
        """
        Compare predicted answer to gold answer.

        Args:
            predicted: Predicted answer
            gold: Gold standard answer
            tolerance: Numeric tolerance (default 1%)

        Returns:
            Dictionary with exact_match, numeric_match, tolerance_match flags
        """
        result = {
            "exact_match": False,
            "numeric_match": False,
            "tolerance_match": False,
        }

        # Normalize strings for comparison
        pred_clean = predicted.strip().lower()
        gold_clean = gold.strip().lower()

        # Exact match
        if pred_clean == gold_clean:
            result["exact_match"] = True
            result["numeric_match"] = True
            result["tolerance_match"] = True
            return result

        # Try numeric comparison
        pred_num = self.extract_number(predicted)
        gold_num = self.extract_number(gold)

        if pred_num is not None and gold_num is not None:
            # Numeric match (exact)
            if abs(pred_num - gold_num) < 1e-6:
                result["numeric_match"] = True
                result["tolerance_match"] = True
            # Tolerance match (within 1%)
            elif gold_num != 0:
                error = abs(pred_num - gold_num) / abs(gold_num)
                if error <= tolerance:
                    result["tolerance_match"] = True

        return result

    def evaluate_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate agent on a single example.

        Args:
            example: FinQA example

        Returns:
            Evaluation result dictionary
        """
        question = example["question"]
        gold_answer = str(example["answer"])
        gold_program = example.get("program", "")

        logger.info("evaluating_example", question=question[:80])

        # Run agent
        try:
            result = self.agent.run(question)

            # Extract metrics
            predicted_answer = result.get("final_answer", "")
            verification_status = result.get("verification_status", "UNCERTAIN")
            verification_confidence = result.get("verification_confidence", "LOW")
            retry_count = result.get("retry_count", 0)
            retry_exhausted = result.get("retry_exhausted", False)

            # Compare answers
            comparison = self.compare_answers(predicted_answer, gold_answer)

            eval_result = {
                "question": question,
                "gold_answer": gold_answer,
                "gold_program": gold_program,
                "predicted_answer": predicted_answer,
                "exact_match": comparison["exact_match"],
                "numeric_match": comparison["numeric_match"],
                "tolerance_match": comparison["tolerance_match"],
                "verification_status": verification_status,
                "verification_confidence": verification_confidence,
                "retry_count": retry_count,
                "retry_exhausted": retry_exhausted,
                "error": None,
            }

        except Exception as e:
            logger.error(
                "evaluation_error",
                question=question[:80],
                error=str(e),
            )
            eval_result = {
                "question": question,
                "gold_answer": gold_answer,
                "gold_program": gold_program,
                "predicted_answer": None,
                "exact_match": False,
                "numeric_match": False,
                "tolerance_match": False,
                "verification_status": "ERROR",
                "verification_confidence": "LOW",
                "retry_count": 0,
                "retry_exhausted": False,
                "error": str(e),
            }

        return eval_result

    def evaluate(self, num_examples: int = 20) -> Dict[str, Any]:
        """
        Evaluate agent on validation set.

        Args:
            num_examples: Number of examples to evaluate

        Returns:
            Evaluation results with metrics
        """
        logger.info("evaluation_started", num_examples=num_examples)

        # Load agent index
        print("\nLoading agent and retriever index...")
        if not self.agent.load_index():
            print("Error: Index not found. Building index first...")
            self.agent.retriever.build_index()
            self.agent.retriever.save_index()
            print("Index built successfully.")

        # Load validation set
        print(f"\nLoading {num_examples} validation examples...")
        val_dataset = load_finqa_dataset(split="validation")

        # HuggingFace Dataset slicing returns dict of lists, convert to list of dicts
        examples_slice = val_dataset[:num_examples]
        num_loaded = len(examples_slice["question"])

        # Convert from dict of lists to list of dicts
        examples = [
            {
                "question": examples_slice["question"][i],
                "answer": examples_slice["answer"][i],
                "program": examples_slice["program"][i],
                "exe_ans": examples_slice["exe_ans"][i],
            }
            for i in range(num_loaded)
        ]

        print(f"\nEvaluating {len(examples)} examples...\n")

        # Evaluate each example
        results = []
        for i, example in enumerate(examples, 1):
            print(f"[{i}/{len(examples)}] Evaluating: {example['question'][:80]}...")
            eval_result = self.evaluate_example(example)
            results.append(eval_result)

            # Print result
            status = "✓" if eval_result["tolerance_match"] else "✗"
            print(f"  {status} Predicted: {eval_result['predicted_answer']}")
            print(f"    Gold: {eval_result['gold_answer']}")
            print()

        # Calculate metrics
        metrics = self.calculate_metrics(results)

        # Create summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_examples": len(results),
            "metrics": metrics,
            "results": results,
        }

        # Save results
        with open(self.output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "evaluation_completed",
            num_examples=len(results),
            output_path=str(self.output_path),
            **metrics
        )

        return summary

    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate aggregate metrics from results.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary of metrics
        """
        total = len(results)
        if total == 0:
            return {}

        # Accuracy metrics
        exact_matches = sum(1 for r in results if r["exact_match"])
        numeric_matches = sum(1 for r in results if r["numeric_match"])
        tolerance_matches = sum(1 for r in results if r["tolerance_match"])

        # Verification metrics
        pass_count = sum(1 for r in results if r["verification_status"] == "PASS")
        fail_count = sum(1 for r in results if r["verification_status"] == "FAIL")
        uncertain_count = sum(1 for r in results if r["verification_status"] == "UNCERTAIN")
        error_count = sum(1 for r in results if r["verification_status"] == "ERROR")

        # Retry metrics
        retry_triggered = sum(1 for r in results if r["retry_count"] > 0)
        total_retries = sum(r["retry_count"] for r in results)
        retry_exhausted = sum(1 for r in results if r["retry_exhausted"])

        # Error rate
        errors = sum(1 for r in results if r["error"] is not None)

        metrics = {
            # Accuracy
            "exact_match_rate": round(exact_matches / total, 3),
            "numeric_match_rate": round(numeric_matches / total, 3),
            "tolerance_match_rate": round(tolerance_matches / total, 3),

            # Verification
            "verification_pass_rate": round(pass_count / total, 3),
            "verification_fail_rate": round(fail_count / total, 3),
            "verification_uncertain_rate": round(uncertain_count / total, 3),

            # Retry
            "retry_triggered_rate": round(retry_triggered / total, 3),
            "avg_retries_per_question": round(total_retries / total, 2),
            "retry_exhausted_rate": round(retry_exhausted / total, 3),

            # Errors
            "error_rate": round(errors / total, 3),
        }

        return metrics

    def print_summary(self, summary: Dict[str, Any]) -> None:
        """
        Print evaluation summary.

        Args:
            summary: Evaluation summary dictionary
        """
        metrics = summary["metrics"]

        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nTimestamp: {summary['timestamp']}")
        print(f"Examples evaluated: {summary['num_examples']}")
        print(f"Results saved to: {self.output_path}")

        print("\n" + "-" * 80)
        print("ACCURACY METRICS")
        print("-" * 80)
        print(f"  Exact match rate:      {metrics['exact_match_rate']:.1%}")
        print(f"  Numeric match rate:    {metrics['numeric_match_rate']:.1%}")
        print(f"  Tolerance match rate:  {metrics['tolerance_match_rate']:.1%} (within 1%)")

        print("\n" + "-" * 80)
        print("VERIFICATION METRICS")
        print("-" * 80)
        print(f"  Pass rate:             {metrics['verification_pass_rate']:.1%}")
        print(f"  Fail rate:             {metrics['verification_fail_rate']:.1%}")
        print(f"  Uncertain rate:        {metrics['verification_uncertain_rate']:.1%}")

        print("\n" + "-" * 80)
        print("RETRY METRICS")
        print("-" * 80)
        print(f"  Retry triggered rate:  {metrics['retry_triggered_rate']:.1%}")
        print(f"  Avg retries/question:  {metrics['avg_retries_per_question']:.2f}")
        print(f"  Retry exhausted rate:  {metrics['retry_exhausted_rate']:.1%}")

        print("\n" + "-" * 80)
        print("ERROR METRICS")
        print("-" * 80)
        print(f"  Error rate:            {metrics['error_rate']:.1%}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate FinQA Agent on validation set")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=20,
        help="Number of validation examples to evaluate (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval_results.json",
        help="Path to save results (default: data/eval_results.json)"
    )
    args = parser.parse_args()

    # Run evaluation
    evaluator = FinQAEvaluator(output_path=args.output)
    summary = evaluator.evaluate(num_examples=args.num_examples)
    evaluator.print_summary(summary)
