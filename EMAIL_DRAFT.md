Subject: FinQA QA Chatbot Take-Home Submission

Hi,

Please find my FinQA take-home assignment materials below:

- GitHub repository: https://github.com/Sirisha0812/runara-finqa
- Technical report: attached / linked from the repository as `TECHNICAL_REPORT.md`
- Presentation materials: `PRESENTATION.md`
- Presentation deck: `outputs/finqa_presentation.pptx`

Summary of the approach:

- Built a Python QA pipeline for FinQA-style numerical reasoning over financial documents.
- Used LangGraph for workflow orchestration, LangChain for model integration, and vLLM as the serving layer for a Hugging Face open-source model.
- Corrected the retrieval design to use document-local evidence rather than unrelated training QA examples.
- Added an evaluation harness that separates answer quality, retrieval quality, and numerical reasoning quality.
- Included production-oriented monitoring and drift considerations.

One note: the repository now includes the complete evaluation harness, but final benchmark numbers should be generated in the target GPU-backed vLLM environment by running:

```bash
python3 quick_eval_and_report.py --split validation --num-examples 100
```

Thank you for the opportunity. I’m happy to walk through the design choices, tradeoffs, and next steps in detail.

Best,
Sirisha
