"""Minimal end-to-end smoke test against a live vLLM server."""

from src.agent import FinQAAgent
from src.data_loader import load_finqa_dataset


def main() -> None:
    example = load_finqa_dataset(split="validation")[0]
    agent = FinQAAgent()
    result = agent.run(example["question"], example)

    print("Question:", example["question"])
    print("Gold answer:", example.get("exe_ans") or example.get("answer"))
    print("Final answer:")
    print(result["final_answer"])
    print("\nRetrieved chunks:", [chunk["chunk_id"] for chunk in result["retrieved_chunks"]])
    print("Verification:", result["verification_status"], result["verification_confidence"])


if __name__ == "__main__":
    main()
