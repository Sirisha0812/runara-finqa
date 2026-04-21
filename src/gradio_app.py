"""Gradio interface for the FinQA chatbot."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, Tuple

try:
    import gradio as gr
except ImportError:  # pragma: no cover - optional runtime dependency
    gr = None

from src.agent import FinQAAgent
from src.data_loader import load_finqa_dataset, prepare_example_for_rag


@lru_cache(maxsize=1)
def get_validation_examples() -> List[Dict[str, Any]]:
    """Load validation examples once for the UI session."""
    return load_finqa_dataset(split="validation")


@lru_cache(maxsize=1)
def get_agent() -> FinQAAgent:
    """Load the chatbot agent once for the UI session."""
    return FinQAAgent()


def build_example_choices(limit: int = 100) -> List[Tuple[str, int]]:
    """Create readable dropdown labels for the first validation examples."""
    dataset = get_validation_examples()
    choices: List[Tuple[str, int]] = []
    for idx, example in enumerate(dataset[:limit]):
        question = example["question"].strip().replace("\n", " ")
        label = f"{idx}: {question[:110]}"
        choices.append((label, idx))
    return choices


def load_example(example_index: int) -> Tuple[str, str, str, str]:
    """Populate the UI with a selected dataset example."""
    example = get_validation_examples()[example_index]
    prepared = prepare_example_for_rag(example)
    gold_answer = str(example.get("exe_ans") or example.get("answer") or "")
    metadata = {
        "example_id": example.get("id"),
        "gold_answer": gold_answer,
        "program": example.get("program", ""),
    }
    return (
        example["question"],
        prepared["table_str"] or "No table for this example.",
        prepared["context"],
        json.dumps(metadata, indent=2),
    )


def run_chatbot(example_index: int, question: str) -> Tuple[str, str, str, str, str]:
    """Run the agent on the selected example and question."""
    example = dict(get_validation_examples()[example_index])
    question = question.strip() or example["question"]
    example["question"] = question

    result = get_agent().run(question, example)

    retrieved = []
    for chunk in result["retrieved_chunks"]:
        retrieved.append(
            f"[{chunk['chunk_id']}] {chunk['chunk_type']} "
            f"(score={chunk['hybrid_score']:.3f})\n{chunk['text']}"
        )

    summary = {
        "verification_status": result["verification_status"],
        "verification_confidence": result["verification_confidence"],
        "retry_count": result["retry_count"],
        "cited_evidence_ids": result.get("cited_evidence_ids", []),
        "calculation_expression": result.get("calculation_expression"),
        "calculation_result": result.get("calculation_result"),
        "issues": result.get("verification_issues", []),
    }

    return (
        result["final_answer"],
        json.dumps(summary, indent=2),
        "\n\n---\n\n".join(retrieved) or "No chunks retrieved.",
        result["reasoning"] or "No reasoning returned.",
        json.dumps(result["node_traces"], indent=2),
    )


def build_app() -> Any:
    """Construct the Gradio app."""
    if gr is None:
        raise ImportError("Gradio is not installed. Run `pip install -r requirements.txt`.")

    default_question, default_table, default_context, default_meta = load_example(0)
    choices = build_example_choices()

    with gr.Blocks(title="FinQA Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # FinQA Numerical Reasoning Chatbot
            Pick a validation example or edit the question, then run the LangGraph agent against the selected financial document.
            """
        )

        with gr.Row():
            example_selector = gr.Dropdown(
                choices=choices,
                value=0,
                label="Validation Example",
            )
            run_button = gr.Button("Run Chatbot", variant="primary")

        question_input = gr.Textbox(
            value=default_question,
            lines=3,
            label="Question",
            placeholder="Ask a FinQA-style numerical reasoning question.",
        )

        with gr.Row():
            metadata_box = gr.Code(
                value=default_meta,
                language="json",
                label="Example Metadata",
            )
            table_box = gr.Textbox(
                value=default_table,
                lines=14,
                label="Table View",
            )

        context_box = gr.Textbox(
            value=default_context,
            lines=10,
            label="Document Context",
        )

        with gr.Row():
            answer_box = gr.Textbox(lines=4, label="Final Answer")
            summary_box = gr.Code(language="json", label="Verification Summary")

        evidence_box = gr.Textbox(lines=14, label="Retrieved Evidence")
        reasoning_box = gr.Textbox(lines=14, label="Reasoning Trace")
        node_trace_box = gr.Code(language="json", label="Node Timing Trace")

        example_selector.change(
            fn=load_example,
            inputs=[example_selector],
            outputs=[question_input, table_box, context_box, metadata_box],
        )

        run_button.click(
            fn=run_chatbot,
            inputs=[example_selector, question_input],
            outputs=[answer_box, summary_box, evidence_box, reasoning_box, node_trace_box],
        )

    return demo


def main() -> None:
    """Launch the Gradio server."""
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
