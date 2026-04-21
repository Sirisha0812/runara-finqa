# FinQA Numerical Reasoning Chatbot

Repository: [github.com/Sirisha0812/runara-finqa](https://github.com/Sirisha0812/runara-finqa)

This project builds a FinQA-style question-answering chatbot for numerical reasoning over financial documents. The implementation uses:

- LangGraph for the execution graph and retry control flow
- LangChain for prompt/model integration
- vLLM as the OpenAI-compatible serving layer for a Hugging Face model
- A document-local hybrid retriever so answers are grounded in the actual source document, not unrelated training QA pairs

## What Changed

The original repository structure was useful, but the retrieval path was methodologically wrong for FinQA: it retrieved other training questions instead of evidence from the target document. The current implementation fixes that by:

- chunking the target financial document into retrievable evidence spans
- retrieving only from those chunks during answering
- keeping train-set usage optional for future few-shot or offline analysis instead of factual evidence lookup
- separating answer accuracy, retrieval quality, and numerical reasoning quality in evaluation

## Architecture

```text
question + financial document
        |
        v
document-local hybrid retrieval
        |
        v
LangGraph state machine
retrieve -> reason -> calculate -> verify -> answer
                   ^              |
                   |______________|
                     retry on FAIL
```

Core files:

- [src/agent.py](/Users/sirishag/Desktop/runara-finqa/src/agent.py)
- [src/retriever.py](/Users/sirishag/Desktop/runara-finqa/src/retriever.py)
- [src/evaluator.py](/Users/sirishag/Desktop/runara-finqa/src/evaluator.py)
- [src/data_loader.py](/Users/sirishag/Desktop/runara-finqa/src/data_loader.py)
- [src/monitoring.py](/Users/sirishag/Desktop/runara-finqa/src/monitoring.py)
- [TECHNICAL_REPORT.md](/Users/sirishag/Desktop/runara-finqa/TECHNICAL_REPORT.md)
- [PRESENTATION.md](/Users/sirishag/Desktop/runara-finqa/PRESENTATION.md)

## Dataset Snapshot

Computed from the local FinQA files in `data/raw`:

- Train examples: 6,251
- Validation examples: 883
- Average context length: about 4,089 chars in train, 4,030 chars in validation
- Average question length: about 16.6 tokens
- Average table rows: 5.3
- Most common gold operations: `divide`, `subtract`, `add`, `multiply`

These stats are produced by [analyze_dataset](/Users/sirishag/Desktop/runara-finqa/src/data_loader.py:290).

## Model Choice

Default model:

- `Qwen/Qwen2.5-7B-Instruct`

Why this model:

- strong open-source instruct baseline
- broadly supported on Hugging Face and vLLM
- realistic fit for a single L4 or A10G GPU
- large enough to benefit from structured reasoning and verification, unlike very small 1.5B models

Recommended production variants:

- `Qwen/Qwen2.5-7B-Instruct` for standard GPU deployment
- `Qwen/Qwen2.5-7B-Instruct-AWQ` if memory pressure matters more than peak quality

Relevant references:

- [FinQA paper](https://arxiv.org/abs/2109.00122)
- [vLLM OpenAI-compatible server docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [LangChain vLLM integration docs](https://python.langchain.com/docs/integrations/chat/vllm/)
- [LangGraph docs](https://langchain-ai.github.io/langgraph/)
- [Qwen2.5-7B-Instruct model card](https://hf.co/Qwen/Qwen2.5-7B-Instruct)

## Setup

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

Example `.env`:

```bash
VLLM_API_BASE=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
VLLM_PORT=8000
AGENT_RETRIEVAL_TOP_K=5
AGENT_MAX_RETRIES=1
```

### 3. Start vLLM

```bash
python3 -m src.vllm_server
```

The launcher targets vLLM’s OpenAI-compatible API and uses `--generation-config vllm`.

## Running

### Inspect dataset stats

```bash
python3 - <<'PY'
from src.data_loader import load_finqa_dataset, analyze_dataset
print(analyze_dataset(load_finqa_dataset("ibm/finqa", "train")))
PY
```

### Ask one validation example

```bash
python3 - <<'PY'
from src.agent import FinQAAgent
from src.data_loader import load_finqa_dataset

example = load_finqa_dataset(split="validation")[0]
agent = FinQAAgent()
result = agent.run(example["question"], example)
print(result["final_answer"])
PY
```

### Launch the Gradio interface

```bash
python3 -m src.gradio_app
```

The UI lets you:

- choose a validation example
- edit the question before running
- inspect the selected table and document context
- view the final answer, verification summary, retrieved evidence, reasoning trace, and node timings

### Run evaluation

```bash
python3 quick_eval_and_report.py --split validation --num-examples 100
```

Outputs:

- `data/eval_results_validation_100.json`
- `data/eval_summary_validation_100.md`

## Hugging Face GPU Provisioning

If you want to provision the serving model on Hugging Face GPU infrastructure instead of a local GPU, use a GPU flavor such as `l4x1`, `a10g-large`, or `a100-large` and run a vLLM container or Python job that starts `vllm serve`.

Recommended serving target:

- Model: `Qwen/Qwen2.5-7B-Instruct`
- GPU: `l4x1` or `a10g-large`
- Tensor parallel: `1`
- Context length: `4096` or higher

The repo code already assumes an OpenAI-compatible endpoint, so switching from local vLLM to a remote vLLM endpoint is just an `.env` change.

## Evaluation Design

The evaluator tracks three groups of metrics:

- Answer correctness: exact match, numeric match, tolerance match
- Retrieval quality: heuristic evidence support at top-k
- Numerical reasoning quality: calculator precision/recall and operator match against the gold FinQA program

That split matters because FinQA failures can come from:

- missing the right table row
- extracting the wrong operands
- performing the wrong arithmetic
- formatting the final answer incorrectly

## Production Readiness

Production hooks already included:

- structured logging with GPU-aware optional metrics
- Prometheus/Grafana scaffolding in `monitoring/`
- drift-detection and maintenance concepts in [src/monitoring.py](/Users/sirishag/Desktop/runara-finqa/src/monitoring.py)
- docker-compose for local monitoring stack

## Deliverables In Repo

- Working implementation: this repository
- Technical report: [TECHNICAL_REPORT.md](/Users/sirishag/Desktop/runara-finqa/TECHNICAL_REPORT.md)
- Presentation materials: [PRESENTATION.md](/Users/sirishag/Desktop/runara-finqa/PRESENTATION.md)

## Current Limitation

This workspace does not currently have the full LLM runtime dependencies or a live vLLM endpoint, so benchmark numbers are not checked into the repo by default. The evaluation harness is complete; run it in the target GPU environment to produce the final quantitative results for submission.
