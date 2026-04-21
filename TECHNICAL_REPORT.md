# FinQA Numerical Reasoning Chatbot

Repository: [https://github.com/Sirisha0812/runara-finqa](https://github.com/Sirisha0812/runara-finqa)

## 1. Objective

The goal is to build a chatbot that answers questions requiring numerical reasoning over financial documents, using the FinQA benchmark as the primary development dataset. The system should be implemented in Python and emphasize:

- Hugging Face model provisioning on GPU
- vLLM serving
- LangChain integration
- LangGraph orchestration
- production-grade monitoring and evaluation thinking

## 2. Dataset Understanding

### 2.1 What FinQA Contains

FinQA pairs:

- a financial document snippet
- associated table content
- a question
- a gold answer
- a gold executable reasoning program

From the local dataset files in this repo:

| Split | Examples | Avg Context Chars | Avg Question Tokens | Avg Table Rows |
|---|---:|---:|---:|---:|
| Train | 6,251 | 4,088.5 | 16.6 | 5.3 |
| Validation | 883 | 4,030.2 | 16.4 | 5.3 |

Gold program operator mix is dominated by arithmetic:

- `divide`: about 46%
- `subtract`: about 29 to 31%
- `add`: about 13 to 16%
- `multiply`: about 6%

The dataset is therefore not plain extractive QA. Nearly every example involves at least one structured reasoning program.

### 2.2 Assumptions

I am making these assumptions:

1. The chatbot operates over a specific financial document at query time, not over a generic web corpus.
2. The right evidence lives inside the target document and its table, so retrieval should be document-local.
3. The system should prefer interpretable arithmetic over implicit model-only reasoning whenever possible.
4. The evaluation should distinguish retrieval failure from reasoning failure.

### 2.3 What Makes Financial QA Unique

Financial QA differs from general-domain QA in several ways:

- Numeric values are dense and easy to confuse across adjacent rows and columns.
- Units matter: percent, millions, billions, basis-point style differences, and fiscal-year qualifiers.
- Questions often require combining table rows with narrative text.
- Small extraction errors cascade into completely wrong arithmetic.
- Hallucination is especially dangerous because wrong financial numbers can still sound plausible.

In short, a strong system must be grounded, numerically explicit, and auditable.

## 3. Method Selection

### 3.1 Approaches Considered

#### Option A: Fine-tuned seq2seq model

Pros:

- close to the original FinQA paper setup
- potentially strong benchmark accuracy
- direct optimization on gold programs/answers

Cons:

- heavier training pipeline
- more compute and experiment time
- harder to adapt quickly to new document formats
- less transparent during debugging unless explicit program supervision is added

#### Option B: Pure prompting over full context

Pros:

- simplest implementation
- no retriever to maintain

Cons:

- context windows are wasted on irrelevant text
- prone to number confusion and hallucination
- hard to scale to larger documents

#### Option C: RAG over all train examples

Pros:

- easy to implement if train examples are already embedded

Cons:

- methodologically weak for FinQA answering
- retrieves similar questions instead of source evidence
- risks leakage and poor factual grounding

This is the main issue I corrected in the repository.

#### Option D: Agentic document-local RAG with calculator and verifier

Pros:

- evidence stays anchored to the active document
- reasoning remains inspectable
- calculator node reduces arithmetic slippage
- verifier node supports self-correction
- LangGraph makes retries and instrumentation clean

Cons:

- more moving parts than a plain prompt
- evaluation has to cover multiple failure modes
- verifier quality still depends on the underlying model

### 3.2 Chosen Approach

I chose Option D: a LangGraph workflow with document-local hybrid retrieval, LLM reasoning, deterministic arithmetic execution, and a verification/retry loop.

### 3.3 Why This Choice Fits FinQA

This design matches the actual structure of the problem:

- retrieval isolates the right evidence spans from the active document
- the LLM composes the reasoning chain
- arithmetic is executed separately instead of trusted blindly
- verification explicitly checks whether the answer is grounded

This is a better production-oriented compromise than either full fine-tuning or a naive prompt-only baseline.

## 4. System Design

### 4.1 Stack

- Model serving: vLLM
- Model wrapper: LangChain `ChatOpenAI` against vLLM’s OpenAI-compatible API
- Orchestration: LangGraph `StateGraph`
- Retrieval: document-local hybrid retrieval
- Arithmetic: `sympy` safe evaluation fallback
- Monitoring: structured logs plus Prometheus/Grafana scaffolding

### 4.2 Model Choice

Default model: `Qwen/Qwen2.5-7B-Instruct`

Rationale:

- stable Hugging Face availability
- good open-source instruct baseline
- realistic deployment target on L4/A10G-class GPUs
- better tradeoff than a tiny 1.5B model for FinQA-style reasoning

Recommended deployment variants:

- standard: `Qwen/Qwen2.5-7B-Instruct`
- memory-optimized: `Qwen/Qwen2.5-7B-Instruct-AWQ`

### 4.3 Retrieval Design

The retriever chunks:

- `pre_text` paragraphs
- table header
- table rows converted into header-value statements
- `post_text` paragraphs

Then it ranks chunks with hybrid scoring:

- dense score if sentence-transformers is installed
- sparse score using BM25 or token-overlap fallback

The important design change is that retrieval is over the current document only, not over unrelated train examples.

### 4.4 LangGraph Flow

```text
retrieve -> reason -> calculate -> verify -> answer
                         ^           |
                         |___________|
                           retry
```

Node roles:

- `retrieve`: get the best evidence spans
- `reason`: produce explicit reasoning, cited evidence IDs, and a candidate arithmetic expression
- `calculate`: safely execute the arithmetic
- `verify`: run an LLM-based verification plus deterministic heuristics
- `answer`: format the final user-facing answer

## 5. Evaluation Strategy

### 5.1 Answer Correctness

Primary metrics:

- exact match
- numeric match
- tolerance match within 1%

Tolerance match is the most important business-facing metric because many financial answers are numeric and may be formatted differently while still being materially correct.

### 5.2 Numerical Reasoning Quality

FinQA provides gold programs, so answer-only accuracy is not enough. I therefore added:

- calculator precision
- calculator recall
- operator match rate
- operator Jaccard similarity

These show whether the model is invoking arithmetic when needed and whether it is using the right type of arithmetic.

### 5.3 Retrieval Quality

The repository includes heuristic evidence-support tracking at top-k. In production I would strengthen this with:

- operand-support recall
- row-level evidence recall
- attribution coverage over cited chunks

### 5.4 Operational Metrics

- average latency
- verification pass/fail/uncertain rates
- retry rate
- retry exhaustion rate

These matter because a numerically cautious system is only useful if it remains fast enough and does not silently degrade into low-confidence answers.

## 6. Results

### 6.1 Results Available in This Workspace

What is directly verified in this workspace:

- dataset profiling is complete
- the corrected architecture is implemented
- the evaluation harness is implemented
- retrieval now uses document-local evidence instead of train-example leakage

### 6.2 Benchmark Status

I did not check in full end-to-end benchmark numbers from this sandbox because the environment does not currently have:

- the full LLM runtime dependencies installed
- a running vLLM endpoint backed by a GPU

That means the final submission step in a GPU environment should be:

```bash
python3 quick_eval_and_report.py --split validation --num-examples 100
```

and then optionally:

```bash
python3 quick_eval_and_report.py --split validation --num-examples 883
```

### 6.3 What I Expect to Learn From Those Runs

The main questions to answer empirically are:

1. Does document-local retrieval materially improve answer quality over the earlier train-example retrieval design?
2. How often does the model invoke arithmetic when the gold program requires it?
3. Does the verifier improve tolerance match rate or mainly add latency?
4. Which operator families are most error-prone: division, subtraction, or multi-step combinations?

## 7. Production Plan

### 7.1 Monitoring

I would monitor four groups of signals:

Quality:

- tolerance match rate on labeled traffic or human-reviewed samples
- verification pass rate
- retry exhaustion rate

Retrieval:

- top-k support for answer operands
- retrieval score drift
- chunk citation coverage

Latency and cost:

- p50/p95/p99 latency
- tokens per request
- GPU utilization and memory

Data drift:

- changes in question wording distribution
- changes in document length and table shape
- sudden growth in unsupported operator patterns

### 7.2 Drift Detection

The repo already includes drift-oriented monitoring scaffolding. In production I would alert on:

- tolerance match rate drop greater than 5 percentage points
- verification fail/uncertain rate jump
- retrieval score collapse
- operator mix shift away from the historical baseline

### 7.3 Maintenance Plan

Short term:

- benchmark 7B vs AWQ vs larger model variants
- add a stronger row-level retrieval metric
- improve unit normalization in answer parsing

Medium term:

- add optional few-shot exemplar retrieval from train questions for prompt guidance
- introduce re-ranking for evidence chunks
- add human review workflow for low-confidence answers

Long term:

- fine-tune a smaller model for program prediction or verifier specialization
- support live document ingestion beyond FinQA
- add online evaluation and model-routing by question complexity

## 8. Conclusion

The final architecture is a better fit for FinQA than the original repository baseline because it fixes the key grounding mistake and aligns the stack with the assignment requirements:

- LangChain for model integration
- LangGraph for orchestration
- vLLM for Hugging Face model serving
- explicit numerical reasoning support
- production monitoring plan

The remaining step is straightforward but environment-dependent: run the completed evaluation harness against a live GPU-backed vLLM server and record the final quantitative results for the submission packet.
