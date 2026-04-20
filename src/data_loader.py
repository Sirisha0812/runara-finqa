"""Data loader for FinQA dataset from HuggingFace."""

from typing import Any, Dict, List, Optional

from datasets import load_dataset
import pandas as pd

from src.logger import get_logger, LoggerContext

logger = get_logger(__name__)


def prepare_example_for_rag(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a FinQA example for RAG by formatting context and table.

    Args:
        example: Raw example from dataset with pre_text, table, post_text, etc.

    Returns:
        Dictionary with question, answer, program, context, table_str, raw_table
    """
    # Extract basic fields
    question = example.get("question", "")
    answer = example.get("answer", "")
    program = example.get("program", "")

    # Get raw table
    raw_table = example.get("table", [])

    # Convert table to markdown format
    table_str = ""
    if raw_table and len(raw_table) > 0:
        # First row is headers
        if len(raw_table) > 0:
            headers = raw_table[0]
            # Create header row
            table_str += "| " + " | ".join(headers) + " |\n"
            # Create separator row
            table_str += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            # Add data rows
            for row in raw_table[1:]:
                table_str += "| " + " | ".join(row) + " |\n"

    # Combine pre_text, table, post_text into context
    pre_text = example.get("pre_text", [])
    post_text = example.get("post_text", [])

    # Join pre_text
    if isinstance(pre_text, list):
        pre_text_str = " ".join(pre_text)
    else:
        pre_text_str = str(pre_text)

    # Join post_text
    if isinstance(post_text, list):
        post_text_str = " ".join(post_text)
    else:
        post_text_str = str(post_text)

    # Create full context
    context = f"{pre_text_str}\n\n{table_str}\n\n{post_text_str}"

    return {
        "question": question,
        "answer": answer,
        "program": program,
        "context": context,
        "table_str": table_str,
        "raw_table": raw_table,
    }


def load_finqa_dataset(
    dataset_name: str = "ibm/finqa",
    split: str = "train",
    cache_dir: Optional[str] = None,
) -> Any:
    """
    Load FinQA dataset from HuggingFace.

    Note: Loading from Parquet files directly as dataset scripts are deprecated.

    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to load (train, validation, test)
        cache_dir: Optional directory to cache dataset

    Returns:
        HuggingFace Dataset object
    """
    with LoggerContext(
        logger,
        "load_finqa_dataset",
        dataset_name=dataset_name,
        split=split,
    ):
        # Load from local JSON files (downloaded from FinQA GitHub repo)
        # HuggingFace dataset scripts are deprecated
        import os
        from pathlib import Path

        project_root = Path(__file__).parent.parent
        data_files = {
            "train": str(project_root / "data/raw/train.json"),
            "validation": str(project_root / "data/raw/dev.json"),
            "test": str(project_root / "data/raw/test.json"),
        }

        # Download if not exists
        if not os.path.exists(data_files[split]):
            logger.info("downloading_dataset_file", split=split)
            import urllib.request
            data_dir = project_root / "data/raw"
            data_dir.mkdir(parents=True, exist_ok=True)

            urls = {
                "train": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
                "validation": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
                "test": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
            }

            urllib.request.urlretrieve(urls[split], data_files[split])
            logger.info("download_complete", split=split, file_path=data_files[split])

        # Load directly from JSON file
        import json
        with open(data_files[split], 'r') as f:
            json_data = json.load(f)

        # Flatten nested 'qa' field for easier access
        flattened_data = []
        for item in json_data:
            flat_item = {
                "pre_text": item.get("pre_text", ""),
                "post_text": item.get("post_text", ""),
                "table": item.get("table", []),
                "table_ori": item.get("table_ori", ""),
                "id": item.get("id", ""),
            }
            # Flatten qa field
            if "qa" in item and item["qa"]:
                flat_item["question"] = item["qa"].get("question", "")
                flat_item["answer"] = item["qa"].get("answer", "")
                flat_item["program"] = item["qa"].get("program", "")
                flat_item["steps"] = str(item["qa"].get("steps", []))  # Convert to string
                flat_item["exe_ans"] = str(item["qa"].get("exe_ans", ""))  # Convert to string
            flattened_data.append(flat_item)

        # Convert to Dataset
        from datasets import Dataset
        dataset = Dataset.from_list(flattened_data)

        logger.info(
            "dataset_loaded",
            dataset_name=dataset_name,
            split=split,
            num_examples=len(dataset),
            features=list(dataset.features.keys()),
        )

        return dataset


def display_examples(
    dataset: Any,
    num_examples: int = 5,
    start_idx: int = 0,
) -> None:
    """
    Display examples from the FinQA dataset in a clean, readable format.

    Args:
        dataset: HuggingFace Dataset object
        num_examples: Number of examples to display
        start_idx: Starting index for examples
    """
    logger.info(
        "displaying_examples",
        num_examples=num_examples,
        start_idx=start_idx,
    )

    # Get the subset of examples
    end_idx = min(start_idx + num_examples, len(dataset))
    examples = dataset[start_idx:end_idx]

    # If single example, wrap in list for consistent iteration
    if not isinstance(examples, dict):
        examples = [examples]
    elif isinstance(examples, dict) and not isinstance(examples.get(list(examples.keys())[0]), list):
        # Single example as dict - convert to list of dicts
        examples = [{k: v for k, v in examples.items()}]
    else:
        # Multiple examples - convert from dict of lists to list of dicts
        num_items = len(examples[list(examples.keys())[0]])
        examples = [
            {key: examples[key][i] for key in examples.keys()}
            for i in range(num_items)
        ]

    print("\n" + "="*80)
    print(f"DISPLAYING {len(examples)} EXAMPLES FROM FINQA DATASET")
    print("="*80 + "\n")

    for idx, example in enumerate(examples, start=start_idx):
        # Prepare example for RAG
        prepared = prepare_example_for_rag(example)

        print(f"\n{'─'*80}")
        print(f"EXAMPLE #{idx}")
        print(f"{'─'*80}\n")

        # Question
        print(f"QUESTION:")
        print(f"  {prepared['question']}\n")

        # Answer
        print(f"ANSWER:")
        print(f"  {prepared['answer']}\n")

        # Program
        print(f"PROGRAM:")
        print(f"  {prepared['program']}\n")

        # Table (markdown format)
        print(f"TABLE (Markdown):")
        if prepared['table_str']:
            for line in prepared['table_str'].split('\n'):
                if line:
                    print(f"  {line}")
        else:
            print("  No table available")
        print()

        # Raw table
        print(f"RAW TABLE:")
        print(f"  {prepared['raw_table']}\n")

        # Context (first 500 chars)
        print(f"CONTEXT (first 500 chars):")
        context_preview = prepared['context'][:500] + "..." if len(prepared['context']) > 500 else prepared['context']
        print(f"  {context_preview}\n")

        # Full context length
        print(f"FULL CONTEXT LENGTH: {len(prepared['context'])} characters\n")

    print("="*80)
    print(f"END OF {len(examples)} EXAMPLES")
    print("="*80 + "\n")

    logger.info(
        "examples_displayed",
        num_examples=len(examples),
    )


def get_dataset_info(dataset: Any) -> Dict[str, Any]:
    """
    Get comprehensive information about the dataset structure.

    Args:
        dataset: HuggingFace Dataset object

    Returns:
        Dictionary with dataset metadata
    """
    logger.info("extracting_dataset_info")

    info = {
        "num_examples": len(dataset),
        "features": list(dataset.features.keys()),
        "feature_types": {
            key: str(dataset.features[key])
            for key in dataset.features.keys()
        },
    }

    # Sample first example to see actual data structure
    if len(dataset) > 0:
        first_example = dataset[0]
        info["sample_keys"] = list(first_example.keys())
        info["sample_structure"] = {
            key: type(value).__name__
            for key, value in first_example.items()
        }

    logger.info(
        "dataset_info_extracted",
        num_examples=info["num_examples"],
        num_features=len(info["features"]),
    )

    return info


def print_dataset_info(dataset: Any) -> None:
    """
    Print comprehensive dataset information in a readable format.

    Args:
        dataset: HuggingFace Dataset object
    """
    info = get_dataset_info(dataset)

    print("\n" + "="*80)
    print("FINQA DATASET INFORMATION")
    print("="*80 + "\n")

    print(f"📊 Total Examples: {info['num_examples']}\n")

    print(f"🔑 Features ({len(info['features'])}):")
    for feature in info['features']:
        print(f"  - {feature}: {info['feature_types'][feature]}")
    print()

    if "sample_structure" in info:
        print(f"📋 Sample Data Types:")
        for key, type_name in info['sample_structure'].items():
            print(f"  - {key}: {type_name}")
        print()

    print("="*80 + "\n")


if __name__ == "__main__":
    """
    Example usage: Load dataset and display examples.
    Run with: python -m src.data_loader
    """
    # Load the dataset
    dataset = load_finqa_dataset(split="train")

    # Print dataset info
    print_dataset_info(dataset)

    # Display first 2 examples
    display_examples(dataset, num_examples=2)
