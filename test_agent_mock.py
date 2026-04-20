"""Test agent on validation set questions with mock LLM."""

import json
from typing import Any, Dict, List

from src.agent import FinQAAgent
from src.data_loader import load_finqa_dataset


class MockLLMResponse:
    """Mock OpenAI-style response."""
    def __init__(self, content: str):
        self.content = content


class MockChoice:
    def __init__(self, content: str):
        self.message = MockLLMResponse(content)


class MockCompletionResponse:
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]


class MockOpenAI:
    """Mock OpenAI client that generates realistic responses."""

    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, model: str, messages: List[Dict], **kwargs) -> MockCompletionResponse:
        """Generate mock response based on system message."""
        system_msg = messages[0]["content"] if messages else ""
        user_msg = messages[1]["content"] if len(messages) > 1 else ""

        # Reasoning node
        if "REASONING:" in system_msg and "CALCULATION:" in system_msg:
            return self._mock_reason(user_msg)

        # Verifier node
        if "VERIFICATION_STATUS:" in system_msg:
            return self._mock_verify(user_msg)

        # Answer node
        if "FINAL_ANSWER:" in system_msg:
            return self._mock_answer(user_msg)

        return MockCompletionResponse("Unable to process request.")

    def _mock_reason(self, user_msg: str) -> MockCompletionResponse:
        """Mock reasoning response."""
        # Extract question
        question = user_msg.split("Question: ")[1].split("\n")[0] if "Question: " in user_msg else ""

        # Check if this is a retry (has "CORRECTION REQUIRED")
        is_retry = "CORRECTION REQUIRED" in user_msg

        # Simulate occasional errors on first attempt (30% chance)
        import random
        if not is_retry and random.random() < 0.3:
            # Introduce an intentional error
            response = f"""REASONING:
To find the answer, I will look at the financial data provided in Document 1.
The value appears to be 150.5 million based on the table.
However, I may have misread the specific year or category.

CALCULATION:
150.5 + 20.3

PRELIMINARY_ANSWER:
The value is approximately 170.8 million."""
        else:
            # Correct response (or corrected after retry)
            response = f"""REASONING:
Based on the retrieved financial documents, I need to extract the specific value requested.
Looking at Document 1, the table shows the relevant financial metric.
The exact figure can be found in the row corresponding to the requested year.

CALCULATION:
NONE

PRELIMINARY_ANSWER:
The value can be directly read from the financial table in the retrieved documents."""

        return MockCompletionResponse(response)

    def _mock_verify(self, user_msg: str) -> MockCompletionResponse:
        """Mock verification response."""
        # Check if reasoning contains calculation
        has_calc = "CALCULATION:" in user_msg and "150.5 + 20.3" in user_msg

        # Fail verification if there's a suspicious calculation
        if has_calc:
            response = """VERIFICATION_STATUS: FAIL
CONFIDENCE: MEDIUM
ISSUES:
- The calculation 150.5 + 20.3 does not match any arithmetic described in the reasoning steps
- The numbers 150.5 and 20.3 cannot be traced back to specific values in the retrieved documents
- The reasoning should extract a direct value rather than performing addition"""
        else:
            # Pass verification
            response = """VERIFICATION_STATUS: PASS
CONFIDENCE: HIGH
ISSUES:
- NONE"""

        return MockCompletionResponse(response)

    def _mock_answer(self, user_msg: str) -> MockCompletionResponse:
        """Mock final answer response."""
        # Extract verification status
        if "FAIL" in user_msg and "max retries" in user_msg:
            response = """FINAL_ANSWER: Unable to determine with high confidence (low confidence — max retries reached)
EXPLANATION: After multiple attempts, the reasoning could not be fully verified against the source documents."""
        elif "PASS" in user_msg and "HIGH" in user_msg:
            response = """FINAL_ANSWER: Value extracted from financial documents
EXPLANATION: The answer was successfully verified against the retrieved financial reports with high confidence."""
        elif "PASS" in user_msg:
            response = """FINAL_ANSWER: Value extracted from financial documents
EXPLANATION: The answer was verified but with moderate confidence due to some uncertainty in the source data."""
        else:
            response = """FINAL_ANSWER: Value needs human review (flagged for human review)
EXPLANATION: The verification process returned uncertain results and requires manual inspection."""

        return MockCompletionResponse(response)


def test_agent_on_validation_set(num_questions: int = 3):
    """Test agent on validation set questions."""
    print("\n" + "=" * 100)
    print(f"TESTING FINQA AGENT ON {num_questions} VALIDATION SET QUESTIONS (MOCK LLM)")
    print("=" * 100)

    # Load validation set
    print("\nLoading validation set...")
    val_dataset = load_finqa_dataset(split="validation")
    print(f"Loaded {len(val_dataset)} validation examples\n")

    # Initialize agent
    print("Initializing agent...")
    agent = FinQAAgent()

    # Load index
    print("Loading retriever index...")
    if not agent.load_index():
        print("ERROR: Index not found. Please build index first.")
        return

    # Replace LLM with mock
    print("Replacing LLM with mock (no vLLM server needed)...\n")
    agent.llm = MockOpenAI()

    # Test on first N questions
    results = []
    for idx in range(min(num_questions, len(val_dataset))):
        example = val_dataset[idx]
        question = example["question"]
        ground_truth_answer = example["answer"]
        ground_truth_program = example["program"]

        print("\n" + "=" * 100)
        print(f"TEST QUESTION #{idx + 1}")
        print("=" * 100)
        print(f"\nQuestion: {question}")
        print(f"Ground Truth Answer: {ground_truth_answer}")
        print(f"Ground Truth Program: {ground_truth_program}")

        # Run agent
        print("\nRunning agent workflow...\n")
        result = agent.run(question)

        # Print full trace
        print("\n" + "─" * 100)
        print("FULL NODE TRACE")
        print("─" * 100)

        for step in result["trace"]:
            node = step.get("node", "?")
            attempt = step.get("attempt", "")
            attempt_label = f" (attempt {attempt})" if attempt else ""

            print(f"\n[{node.upper()}{attempt_label}]")

            if node == "retrieve":
                print(f"  • Documents retrieved: {step['num_docs']}")
                print(f"  • Top hybrid score: {step.get('top_hybrid_score', 'N/A'):.4f}")
                print(f"  • Retrieved questions:")
                for q in step.get("retrieved_questions", [])[:2]:  # Show first 2
                    print(f"    - {q}")

            elif node == "reason":
                print(f"  • Is retry: {step.get('is_retry', False)}")
                print(f"  • Calculation expr: {step.get('calculation_expression') or 'NONE'}")
                print(f"  • Reasoning preview:")
                preview = step.get("reasoning_preview", "")
                for line in preview[:300].split("\n")[:10]:  # First 300 chars, 10 lines
                    print(f"    {line}")

            elif node == "calculator":
                if step.get("skipped"):
                    print(f"  • Skipped: {step.get('reason', '')}")
                else:
                    print(f"  • Expression: {step.get('expression')}")
                    print(f"  • Result: {step.get('result')}")

            elif node == "verifier":
                status = step.get("status", "?")
                print(f"  • Status: {status}")
                print(f"  • Confidence: {step.get('confidence', 'N/A')}")
                print(f"  • Retry exhausted: {step.get('retry_exhausted', False)}")
                issues = step.get("issues", [])
                if issues:
                    print(f"  • Issues detected:")
                    for issue in issues:
                        print(f"    ✗ {issue}")
                else:
                    print(f"  • Issues detected: none")

            elif node == "answer":
                print(f"  • Answer mode: {step.get('answer_mode', 'N/A')}")
                print(f"  • Final answer: {step.get('final_answer', '')}")

        # Print node execution timeline
        print("\n" + "─" * 100)
        print("NODE EXECUTION TIMELINE")
        print("─" * 100)
        print(f"{'Node':<12} {'Attempt':<9} {'Duration':>12}    {'Status':<12}")
        print("─" * 100)
        total_ms = 0.0
        for nt in result["node_traces"]:
            attempt_col = str(nt["attempt"]) if "attempt" in nt else "-"
            dur = nt["duration_ms"]
            total_ms += dur
            status = nt.get("status", "ok")
            print(f"{nt['node']:<12} {attempt_col:<9} {dur:>11.2f}ms    {status:<12}")
        print("─" * 100)
        print(f"{'TOTAL':<12} {'':<9} {total_ms:>11.2f}ms")

        # Print summary
        print("\n" + "─" * 100)
        print("SUMMARY")
        print("─" * 100)
        print(f"Retries used: {result['retry_count']} / 2")
        print(f"Retry exhausted: {result['retry_exhausted']}")
        print(f"Verification status: {result['verification_status']}")
        print(f"Verification confidence: {result['verification_confidence']}")
        if result["verification_issues"]:
            print(f"Verification issues:")
            for issue in result["verification_issues"]:
                print(f"  ✗ {issue}")
        print(f"\nFinal Answer: {result['final_answer']}")
        print(f"\nGround Truth: {ground_truth_answer}")

        # Store result
        results.append({
            "question_idx": idx,
            "question": question,
            "ground_truth_answer": ground_truth_answer,
            "final_answer": result["final_answer"],
            "retry_count": result["retry_count"],
            "verification_status": result["verification_status"],
            "verification_confidence": result["verification_confidence"],
            "total_time_ms": total_ms,
        })

    # Print aggregate statistics
    print("\n\n" + "=" * 100)
    print("AGGREGATE STATISTICS")
    print("=" * 100)

    total_retries = sum(r["retry_count"] for r in results)
    avg_time = sum(r["total_time_ms"] for r in results) / len(results)

    verification_statuses = {}
    for r in results:
        status = r["verification_status"]
        verification_statuses[status] = verification_statuses.get(status, 0) + 1

    print(f"\nTotal questions tested: {len(results)}")
    print(f"Total retries triggered: {total_retries}")
    print(f"Average time per question: {avg_time:.2f}ms")
    print(f"\nVerification status breakdown:")
    for status, count in verification_statuses.items():
        print(f"  {status}: {count}")

    print("\n" + "=" * 100)
    print("TEST COMPLETE")
    print("=" * 100)

    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    import random
    random.seed(42)

    results = test_agent_on_validation_set(num_questions=3)
