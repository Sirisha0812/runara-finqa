"""
FinQA Agent Evaluator

Evaluates agent performance on validation set examples.
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
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
        self.agent = None  # Lazy init

    def _init_agent(self):
        """Initialize agent and load index."""
        if self.agent is None:
            print("\nInitializing agent and loading retriever index...")
            self.agent = FinQAAgent()
            if not self.agent.load_index():
                print("Error: Index not found. Building index first...")
                self.agent.retriever.build_index()
                self.agent.retriever.save_index()
                print("Index built successfully.")
            else:
                print("Index loaded successfully.")

    def normalize_string(self, text: str) -> str:
        """
        Normalize string for comparison.

        Args:
            text: Input text

        Returns:
            Normalized string (lowercase, stripped, no $, %, commas)
        """
        if not text:
            return ""
        # Remove $ % , and whitespace, convert to lowercase
        normalized = text.lower().strip()
        normalized = re.sub(r'[\$%,]', '', normalized)
        return normalized

    def extract_number(self, text: str) -> Optional[float]:
        """
        Extract numeric value from text.

        Handles formats like: "45.0", "$45M", "45%", "-45.0", etc.

        Args:
            text: Text containing number

        Returns:
            Extracted number or None
        """
        if not text:
            return None

        # Normalize
        text = self.normalize_string(text)

        # Try to find a number (including negative and decimals)
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group())
            except (ValueError, AttributeError):
                return None

        return None

    def extract_final_answer(self, agent_output: str) -> str:
        """
        Extract final answer from agent output.

        Looks for "FINAL_ANSWER: <answer>" pattern.

        Args:
            agent_output: Full agent output string

        Returns:
            Extracted answer or full output if pattern not found
        """
        if not agent_output:
            return ""

        # Try to find "FINAL_ANSWER: <value>" pattern
        match = re.search(r'FINAL_ANSWER:\s*(.+?)(?:\n|$)', agent_output, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # If no pattern found, return full output
        return agent_output.strip()

    def compare_answers(
        self,
        predicted: str,
        gold: Union[str, float],
        tolerance: float = 0.01
    ) -> Dict[str, bool]:
        """
        Compare predicted answer to gold answer.

        Args:
            predicted: Predicted answer string
            gold: Gold standard answer (str or float)
            tolerance: Numeric tolerance (default 1%)

        Returns:
            Dictionary with exact_match, numeric_match, tolerance_match flags
        """
        result = {
            "exact_match": False,
            "numeric_match": False,
            "tolerance_match": False,
        }

        # Convert gold to string if needed
        gold_str = str(gold)

        # Extract final answer if it's in agent format
        predicted_clean = self.extract_final_answer(predicted)

        # Exact match (normalized strings)
        pred_norm = self.normalize_string(predicted_clean)
        gold_norm = self.normalize_string(gold_str)

        if pred_norm == gold_norm:
            result["exact_match"] = True
            result["numeric_match"] = True
            result["tolerance_match"] = True
            return result

        # Try numeric comparison
        pred_num = self.extract_number(predicted_clean)
        gold_num = self.extract_number(gold_str)

        if pred_num is not None and gold_num is not None:
            # Numeric match (exact within floating point precision)
            if abs(pred_num - gold_num) < 1e-6:
                result["numeric_match"] = True
                result["tolerance_match"] = True
            # Tolerance match (within 1%)
            else:
                denominator = max(abs(gold_num), 1e-8)  # Avoid division by zero
                error = abs(pred_num - gold_num) / denominator
                if error <= tolerance:
                    result["tolerance_match"] = True

        return result

    def evaluate_example(self, example: Dict[str, Any], example_idx: int) -> Dict[str, Any]:
        """
        Evaluate agent on a single example.

        Args:
            example: FinQA example dict
            example_idx: Example index for logging

        Returns:
            Evaluation result dictionary
        """
        # Extract fields
        question = example.get("question", "")
        gold_answer = example.get("answer", "")
        gold_program = example.get("program", "")
        gold_exe_ans = example.get("exe_ans", "")

        logger.info("evaluating_example", idx=example_idx, question=question[:80])

        # Run agent
        try:
            start_time = time.time()
            result = self.agent.run(question)
            latency_ms = (time.time() - start_time) * 1000

            # Extract metrics from agent result
            predicted_answer = result.get("final_answer", "")
            verification_status = result.get("verification_status", "UNCERTAIN")
            verification_confidence = result.get("verification_confidence", "LOW")
            retry_count = result.get("retry_count", 0)
            retry_exhausted = result.get("retry_exhausted", False)

            # Check if calculator was used
            calculator_used = False
            for trace in result.get("trace", []):
                if trace.get("node") == "calculator" and not trace.get("skipped"):
                    calculator_used = True
                    break

            # Compare answers (use exe_ans if available, else answer)
            gold_for_comparison = gold_exe_ans if gold_exe_ans else gold_answer
            comparison = self.compare_answers(predicted_answer, gold_for_comparison)

            eval_result = {
                "example_idx": example_idx,
                "question": question,
                "gold_answer": str(gold_answer),
                "gold_program": gold_program,
                "gold_exe_ans": str(gold_exe_ans),
                "predicted_answer": predicted_answer,
                "exact_match": comparison["exact_match"],
                "numeric_match": comparison["numeric_match"],
                "tolerance_match": comparison["tolerance_match"],
                "verification_status": verification_status,
                "verification_confidence": verification_confidence,
                "retry_count": int(retry_count),
                "retry_exhausted": bool(retry_exhausted),
                "calculator_used": calculator_used,
                "latency_ms": round(latency_ms, 2),
                "error": None,
            }

            logger.info(
                "example_evaluated",
                idx=example_idx,
                tolerance_match=comparison["tolerance_match"],
                retries=retry_count,
                latency_ms=round(latency_ms, 2)
            )

        except Exception as e:
            logger.error(
                "evaluation_error",
                idx=example_idx,
                question=question[:80],
                error=str(e),
            )
            eval_result = {
                "example_idx": example_idx,
                "question": question,
                "gold_answer": str(gold_answer),
                "gold_program": gold_program,
                "gold_exe_ans": str(gold_exe_ans),
                "predicted_answer": None,
                "exact_match": False,
                "numeric_match": False,
                "tolerance_match": False,
                "verification_status": "ERROR",
                "verification_confidence": "LOW",
                "retry_count": 0,
                "retry_exhausted": False,
                "calculator_used": False,
                "latency_ms": 0.0,
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

        # Initialize agent
        self._init_agent()

        # Load validation set
        print(f"\nLoading {num_examples} validation examples...")
        val_dataset = load_finqa_dataset(split="validation")

        # Get individual examples
        # HuggingFace Dataset: iterate directly or use select()
        examples = []
        for i in range(min(num_examples, len(val_dataset))):
            example = val_dataset[i]
            examples.append(example)

        print(f"Loaded {len(examples)} examples from validation set")

        # Debug: Check first example structure
        if len(examples) > 0:
            print(f"\nDebug - First example type: {type(examples[0])}")
            print(f"Debug - First example keys: {examples[0].keys()}")
            print(f"Debug - Sample question: {examples[0]['question'][:80]}...")

        # Run a single test example first
        if len(examples) > 0:
            print("\nRunning test on first example...")
            try:
                test_result = self.evaluate_example(examples[0], 0)
                print(f"✓ Test passed - predicted: {test_result['predicted_answer']}")
                print(f"  Gold: {test_result['gold_answer']}")
                print(f"  Match: {test_result['tolerance_match']}")
            except Exception as e:
                print(f"✗ Test failed: {e}")
                raise

        print(f"\nEvaluating {len(examples)} examples...\n")

        # Evaluate each example
        results = []
        for i, example in enumerate(examples):
            print(f"[{i+1}/{len(examples)}] {example['question'][:80]}...")

            eval_result = self.evaluate_example(example, i)
            results.append(eval_result)

            # Print result
            status = "✓" if eval_result["tolerance_match"] else "✗"
            print(f"  {status} Predicted: {eval_result['predicted_answer']}")
            print(f"    Gold: {eval_result['gold_answer']}")
            if eval_result["error"]:
                print(f"    Error: {eval_result['error']}")
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
        print(f"\nSaving results to {self.output_path}...")
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

        # Calculator usage
        calculator_used = sum(1 for r in results if r["calculator_used"])

        # Latency
        valid_latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
        avg_latency = sum(valid_latencies) / len(valid_latencies) if valid_latencies else 0

        # Error rate
        errors = sum(1 for r in results if r["error"] is not None)

        metrics = {
            # Accuracy
            "exact_match_rate": round(exact_matches / total, 4),
            "numeric_match_rate": round(numeric_matches / total, 4),
            "tolerance_match_rate": round(tolerance_matches / total, 4),

            # Verification
            "verification_pass_rate": round(pass_count / total, 4),
            "verification_fail_rate": round(fail_count / total, 4),
            "verification_uncertain_rate": round(uncertain_count / total, 4),
            "verification_error_rate": round(error_count / total, 4),

            # Retry
            "retry_trigger_rate": round(retry_triggered / total, 4),
            "avg_retries_per_question": round(total_retries / total, 3),
            "retry_exhaustion_rate": round(retry_exhausted / total, 4),

            # Calculator
            "calculator_usage_rate": round(calculator_used / total, 4),

            # Performance
            "avg_latency_ms": round(avg_latency, 2),

            # Errors
            "error_rate": round(errors / total, 4),
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
        print("FINQA AGENT EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nTimestamp: {summary['timestamp']}")
        print(f"Examples evaluated: {summary['num_examples']}")
        print(f"Results saved to: {self.output_path}")

        print("\n" + "-" * 80)
        print("ACCURACY METRICS")
        print("-" * 80)
        print(f"  Exact match rate:      {metrics['exact_match_rate']:>7.1%}")
        print(f"  Numeric match rate:    {metrics['numeric_match_rate']:>7.1%}")
        print(f"  Tolerance match rate:  {metrics['tolerance_match_rate']:>7.1%} (within 1%)")

        print("\n" + "-" * 80)
        print("VERIFICATION METRICS")
        print("-" * 80)
        print(f"  Pass rate:             {metrics['verification_pass_rate']:>7.1%}")
        print(f"  Fail rate:             {metrics['verification_fail_rate']:>7.1%}")
        print(f"  Uncertain rate:        {metrics['verification_uncertain_rate']:>7.1%}")
        print(f"  Error rate:            {metrics['verification_error_rate']:>7.1%}")

        print("\n" + "-" * 80)
        print("RETRY METRICS")
        print("-" * 80)
        print(f"  Retry trigger rate:    {metrics['retry_trigger_rate']:>7.1%}")
        print(f"  Avg retries/question:  {metrics['avg_retries_per_question']:>7.2f}")
        print(f"  Retry exhaustion rate: {metrics['retry_exhaustion_rate']:>7.1%}")

        print("\n" + "-" * 80)
        print("PERFORMANCE METRICS")
        print("-" * 80)
        print(f"  Calculator usage rate: {metrics['calculator_usage_rate']:>7.1%}")
        print(f"  Avg latency:           {metrics['avg_latency_ms']:>7.0f} ms")
        print(f"  Error rate:            {metrics['error_rate']:>7.1%}")

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
