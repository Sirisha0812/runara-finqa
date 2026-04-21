# FinQA Chatbot Presentation

Target length: 30 to 45 minutes

## Slide 1: Title

FinQA Numerical Reasoning Chatbot  
LangGraph + LangChain + vLLM + Hugging Face

Speaker notes:

- Briefly state the assignment goal.
- Position the talk around design choices, not just a code walkthrough.

## Slide 2: Problem Statement

- Answer financial questions that require numerical reasoning
- Work over long financial text plus tables
- Keep answers auditable and production-friendly

Speaker notes:

- Emphasize that this is not standard extractive QA.
- Wrong numbers are high-cost failures in this domain.

## Slide 3: Why FinQA Is Hard

- Questions often require arithmetic, not direct span extraction
- Evidence is split between prose and tables
- Units and time periods matter
- Plausible hallucinations are dangerous

Speaker notes:

- Give a simple example: computing a percentage change from two rows in a table.

## Slide 4: Dataset Snapshot

- Train: 6,251 examples
- Validation: 883 examples
- Avg context length: about 4k characters
- Avg table rows: 5.3
- Dominant gold operators: divide, subtract, add, multiply

Speaker notes:

- Mention that nearly all examples encode structured reasoning programs.

## Slide 5: Initial Design Problem Found

- Original repo retrieved similar train QA examples
- That is not the right evidence source for FinQA answering
- It weakens grounding and makes evaluation hard to trust

Speaker notes:

- This is the most important correction in the project.

## Slide 6: Final System Architecture

```text
question + document
        |
        v
document-local hybrid retrieval
        |
        v
retrieve -> reason -> calculate -> verify -> answer
                         ^           |
                         |___________|
                           retry
```

Speaker notes:

- Stress that retrieval is scoped to the active document.

## Slide 7: Why This Method

- Better grounding than pure prompting
- Lower implementation cost than fine-tuning
- More production-friendly than opaque end-to-end seq2seq
- Easier debugging through explicit node traces

Speaker notes:

- Mention the alternatives considered: fine-tuning, prompt-only, train-example retrieval.

## Slide 8: Retrieval Strategy

- Chunk `pre_text`, table header, table rows, `post_text`
- Score chunks with hybrid retrieval
- Prefer document-local evidence over global corpus lookup
- Keep chunk IDs for traceability

Speaker notes:

- Explain why table rows are converted to header-value statements.

## Slide 9: LangChain + vLLM Integration

- `ChatOpenAI` points at vLLM’s OpenAI-compatible endpoint
- Model default: `Qwen/Qwen2.5-7B-Instruct`
- Same code works locally or against a remote vLLM deployment

Speaker notes:

- Call out that this is a pragmatic integration path with low switching cost.

## Slide 10: LangGraph Design

- `retrieve`: collect evidence
- `reason`: generate explicit reasoning and candidate arithmetic
- `calculate`: execute safely
- `verify`: LLM plus deterministic checks
- `answer`: final concise response

Speaker notes:

- Explain why LangGraph is warranted here: retries, observability, extensibility.

## Slide 11: Numerical Reasoning Controls

- Arithmetic is separated from free-form generation
- Unsafe expressions are blocked
- Verification checks unsupported numbers and answer/result mismatch
- Retry loop triggers on failed verification

Speaker notes:

- This is the core guardrail against plausible but wrong finance answers.

## Slide 12: Model Provisioning Plan

- Default model: `Qwen/Qwen2.5-7B-Instruct`
- Suggested GPU: L4 or A10G
- vLLM endpoint can be local or Hugging Face-hosted
- AWQ variant is an option when memory is constrained

Speaker notes:

- Explain the tradeoff: 7B is the practical baseline, larger models are future work.

## Slide 13: Evaluation Framework

- Answer metrics: exact, numeric, tolerance match
- Retrieval metric: evidence support at top-k
- Reasoning metrics: calculator precision/recall, operator match
- Workflow metrics: latency, verification outcomes, retries

Speaker notes:

- Separate retrieval, reasoning, and formatting failures.

## Slide 14: What I Verified In This Repo

- Dataset analysis runs locally
- Architecture corrected to document-local retrieval
- Evaluation harness implemented
- Monitoring scaffold included

Speaker notes:

- Be explicit that full end-to-end model benchmarking needs the GPU serving environment.

## Slide 15: Expected Demo Flow

1. Start vLLM
2. Load one validation example
3. Show retrieved chunks
4. Show reasoning and arithmetic
5. Show verifier outcome
6. Show final answer

Speaker notes:

- Keep the demo focused on traceability, not flashy UI.

## Slide 16: Production Monitoring

- Quality: tolerance match, verification pass, retry exhaustion
- Performance: p50/p95 latency, GPU memory, throughput
- Drift: question distribution, retrieval score changes, operator mix changes

Speaker notes:

- Mention Prometheus and Grafana assets already in the repo.

## Slide 17: Risks and Limitations

- Final benchmark numbers still need a GPU-backed run
- Retrieval heuristic metrics need stronger evidence labeling
- Small models can still fail on multi-step reasoning
- Verifier quality is only as strong as the base model

Speaker notes:

- This slide matters. It shows judgment rather than overclaiming.

## Slide 18: Next Steps

- Run 100-example and full-validation benchmarks
- Compare 7B vs AWQ vs larger models
- Add re-ranking and stronger unit normalization
- Consider fine-tuning for program prediction later

Speaker notes:

- Close with a concrete roadmap, not vague “future work”.

## Slide 19: Conclusion

- Built a FinQA-specific reasoning pipeline
- Corrected the key grounding flaw in the original repo
- Integrated LangChain, LangGraph, and vLLM cleanly
- Left the project in a state ready for GPU benchmarking and submission

Speaker notes:

- End by restating the most defensible value: correctness, traceability, and production readiness.

## Appendix Demo Questions

- What was the percentage cumulative total return for Citi common stock for the five year period ended December 31, 2017?
- What percentage of total oil and gas MMBOE comes from Canada?
- What was the change in operating income from 2016 to 2017?
