"""Local retrieval-only demo of the FinQA architecture."""

from src.data_loader import load_finqa_dataset
from src.retriever import FinQARetriever


def main() -> None:
    example = load_finqa_dataset(split="validation")[0]
    retriever = FinQARetriever()
    results = retriever.retrieve_for_example(example["question"], example, k=5)

    print("=" * 80)
    print("FINQA DOCUMENT-LOCAL RETRIEVAL DEMO")
    print("=" * 80)
    print("Question:", example["question"])
    print()

    for idx, item in enumerate(results, start=1):
        print(f"{idx}. {item['chunk_id']} [{item['chunk_type']}] score={item['hybrid_score']:.3f}")
        print(f"   {item['text'][:240]}")
        print()

    print("Next step in the full pipeline: send these chunks into the LangGraph agent.")


if __name__ == "__main__":
    main()
