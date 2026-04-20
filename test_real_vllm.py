"""Test agent on specific FinQA validation questions with real vLLM."""

from src.agent import FinQAAgent
from src.data_loader import load_finqa_dataset


def find_question_by_text(dataset, question_text: str):
    """Find a question in the dataset by partial text match."""
    question_lower = question_text.lower()
    for idx, example in enumerate(dataset):
        if question_lower in example["question"].lower():
            return idx, example
    return None, None


def print_separator(char="=", length=100):
    """Print a separator line."""
    print("\n" + char * length)


def print_node_trace(result):
    """Print detailed node trace."""
    print_separator("─")
    print("FULL NODE TRACE")
    print_separator("─")

    for step in result["trace"]:
        node = step.get("node", "?")
        attempt = step.get("attempt", "")
        attempt_label = f" (attempt {attempt})" if attempt else ""

        print(f"\n[{node.upper()}{attempt_label}]")

        if node == "retrieve":
            print(f"  • Documents retrieved: {step['num_docs']}")
            print(f"  • Top hybrid score: {step.get('top_hybrid_score', 'N/A'):.4f}")
            print(f"  • Retrieved questions:")
            for q in step.get("retrieved_questions", []):
                print(f"    - {q}")

        elif node == "reason":
            print(f"  • Is retry: {step.get('is_retry', False)}")
            print(f"  • Calculation expr: {step.get('calculation_expression') or 'NONE'}")
            print(f"\n  • Full Reasoning:")
            reasoning = step.get("reasoning_preview", "")
            # Print full reasoning, not just preview
            for line in reasoning.split("\n"):
                print(f"    {line}")

        elif node == "calculator":
            if step.get("skipped"):
                print(f"  • Skipped: {step.get('reason', '')}")
            else:
                print(f"  • Expression: {step.get('expression')}")
                print(f"  • Result: {step.get('result')}")

        elif node == "verifier":
            status = step.get("status", "?")
            icon = {"PASS": "✓", "FAIL": "✗", "UNCERTAIN": "~"}.get(status, "?")
            print(f"  • Status: {icon} {status}")
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
            print(f"\n  • Final Answer:")
            for line in step.get("final_answer", "").split("\n"):
                print(f"    {line}")


def print_timeline(result):
    """Print node execution timeline."""
    print_separator("─")
    print("NODE EXECUTION TIMELINE")
    print_separator("─")
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


def print_comparison(question, result, gold_answer, gold_program):
    """Print comparison between final answer and gold answer."""
    print_separator("─")
    print("ANSWER COMPARISON")
    print_separator("─")
    print(f"\nQuestion: {question}")
    print(f"\nRetries used: {result['retry_count']} / 2")
    print(f"Retry exhausted: {result['retry_exhausted']}")
    print(f"Verification status: {result['verification_status']}")
    print(f"Verification confidence: {result['verification_confidence']}")

    if result["verification_issues"]:
        print(f"\nVerification issues:")
        for issue in result["verification_issues"]:
            print(f"  ✗ {issue}")

    print(f"\n{'─'*100}")
    print(f"AGENT ANSWER:\n{result['final_answer']}")
    print(f"\n{'─'*100}")
    print(f"GOLD ANSWER: {gold_answer}")
    print(f"GOLD PROGRAM: {gold_program}")

    if result.get("calculation_result"):
        print(f"\nAgent Calculation: {result['calculation_expression']} = {result['calculation_result']}")


def main():
    """Run agent on 3 specific FinQA validation questions."""
    print_separator()
    print("TESTING AGENT ON 3 FINQA VALIDATION QUESTIONS (REAL vLLM)")
    print_separator()

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

    print("\n✅ Agent ready with real vLLM server\n")

    # Test questions
    questions = [
        "what is the interest expense in 2009?",
        "what was the percentage cumulative total return for citi common stock for the five year period ended december 31 2017?",
        "what percentage of total oil and gas mmboe comes from canada?",
    ]

    for q_num, question_text in enumerate(questions, 1):
        print_separator("=")
        print(f"TEST QUESTION #{q_num}")
        print_separator("=")

        # Find question in dataset
        idx, example = find_question_by_text(val_dataset, question_text)

        if example is None:
            print(f"\n⚠️  Question not found in validation set: {question_text}")
            print("Using question text directly without gold answer...")
            question = question_text
            gold_answer = "N/A"
            gold_program = "N/A"
        else:
            question = example["question"]
            gold_answer = example["answer"]
            gold_program = example["program"]

        print(f"\nQuestion: {question}")
        print(f"Gold Answer: {gold_answer}")
        print(f"Gold Program: {gold_program}")

        # Run agent
        print("\nRunning agent workflow...\n")
        result = agent.run(question)

        # Print detailed trace
        print_node_trace(result)
        print_timeline(result)
        print_comparison(question, result, gold_answer, gold_program)

    print_separator("=")
    print("ALL TESTS COMPLETE")
    print_separator("=")


if __name__ == "__main__":
    main()
