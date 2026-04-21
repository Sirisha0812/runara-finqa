"""Simple retrieval sanity checks for a few validation examples."""

from src.data_loader import load_finqa_dataset
from src.retriever import FinQARetriever


def main() -> None:
    dataset = load_finqa_dataset(split="validation")
    retriever = FinQARetriever()

    for example in dataset[:3]:
        print("=" * 100)
        print("Question:", example["question"])
        print("Gold answer:", example.get("exe_ans") or example.get("answer"))
        results = retriever.retrieve_for_example(example["question"], example, k=5)
        for item in results:
            print(
                f"- {item['chunk_id']} [{item['chunk_type']}] "
                f"hybrid={item['hybrid_score']:.3f}: {item['text'][:140]}"
            )


if __name__ == "__main__":
    main()
