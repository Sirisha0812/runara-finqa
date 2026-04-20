# FinQA Project - Complete File Structure & Explanation

## 📂 Project Overview

**Purpose**: Financial question-answering agent using LangGraph + vLLM + Hybrid Retrieval
**Tech Stack**: Python, vLLM, LangGraph, LangChain, FAISS, BM25
**Dataset**: IBM Research FinQA (6,251 training examples)

---

## 📁 File-by-File Explanation

### **🔧 Configuration Files**

#### `.env` (gitignored)
**Purpose**: Your local environment variables
**Contains**:
- vLLM API endpoint (from Colab: `https://8000-gpu-t4...`)
- Model name (`Qwen/Qwen2.5-1.5B-Instruct`)
- Embedding model, logging config

**Why**: Keeps secrets and local config out of git

---

#### `.env.example`
**Purpose**: Template for environment variables
**Lines**: 25
**Contains**:
```bash
VLLM_API_BASE=http://localhost:8000/v1
VLLM_MODEL=Qwen/Qwen2.5-32B-Instruct
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**Why**: Shows others what env vars are needed (safe to commit)

---

#### `.gitignore`
**Purpose**: Tell git which files to ignore
**Ignores**:
- `.env` (secrets)
- `__pycache__/` (Python bytecode)
- `*.pyc` (compiled Python)

**Why**: Prevent committing secrets or generated files

---

#### `requirements.txt`
**Purpose**: Python dependencies
**Lines**: 39
**Key Dependencies**:
- `vllm>=0.6.0` - GPU-accelerated LLM serving
- `langgraph>=0.2.0` - Workflow orchestration
- `langchain>=0.3.0` - RAG components
- `faiss-cpu>=1.7.4` - Vector search
- `rank-bm25>=0.2.2` - Keyword search
- `sentence-transformers>=2.2.0` - Embeddings
- `structlog>=23.1.0` - Structured logging

**Install**: `pip install -r requirements.txt`

**Why**: Reproducible environment for anyone cloning the repo

---

### **📝 Documentation Files**

#### `README.md`
**Purpose**: Main project documentation
**Lines**: 150+
**Sections**:
- Project overview
- Architecture diagram
- Quick start guide
- Features list
- Installation instructions

**Audience**: Anyone discovering the project

**Why**: First thing people see on GitHub

---

#### `CLAUDE.md`
**Purpose**: Development context and decisions
**Lines**: 120+
**Contains**:
- Todo list (what's done, what's next)
- Design decisions (why hybrid retrieval? why LangGraph?)
- Configuration choices
- Development notes

**Audience**: You (to remember context) and future developers

**Why**: Documents the "why" behind technical choices

---

#### `CODE_EXPLAIN.md`
**Purpose**: Line-by-line code explanations
**Lines**: Extensive
**Explains**:
- How each function works
- Why certain approaches were chosen
- Code patterns and best practices

**Audience**: Developers learning the codebase

**Why**: Makes onboarding easier, shows deep understanding

---

#### `COLAB_SETUP.md`
**Purpose**: Google Colab deployment guide
**Lines**: 180+
**Contains**:
- Step-by-step Colab setup
- vLLM server startup commands
- Troubleshooting tips
- Expected output examples

**Audience**: Anyone running this on Colab (like you did!)

**Why**: Makes GPU deployment reproducible

---

#### `DEPLOYMENT_STRATEGY.md`
**Purpose**: Production deployment guide
**Lines**: 150+
**Contains**:
- Why vLLM matters for Runara interview
- Cloud GPU options (RunPod, Lambda, Vast.ai)
- Cost estimates
- Performance optimization tips

**Audience**: Interview reviewers, production deployers

**Why**: Shows production thinking (critical for Runara!)

---

#### `RESULTS.md`
**Purpose**: Test results and performance analysis
**Lines**: 276
**Contains**:
- Full test execution breakdown
- GPU metrics (61% utilization, 91.6% memory)
- Performance timings (11.2s total)
- Production recommendations

**Audience**: Interview reviewers, stakeholders

**Why**: Proves the system works with real data

---

### **💻 Source Code (`src/` directory)**

#### `src/__init__.py`
**Purpose**: Makes `src` a Python package
**Lines**: 0 (empty file)

**Why**: Allows `from src.agent import FinQAAgent`

---

#### `src/config.py`
**Purpose**: Configuration management with Pydantic
**Lines**: 121
**Classes**:
- `VLLMConfig` - vLLM server settings
- `LoggingConfig` - Log level, format
- `VectorStoreConfig` - Embedding model, paths
- `PerformanceConfig` - GPU monitoring, retrieval settings
- `Config` - Main config class (loads from .env)

**Key Features**:
- Type-safe configuration
- Automatic .env loading
- Validation (e.g., log level must be INFO/DEBUG/ERROR)

**Usage**:
```python
from src.config import config
print(config.vllm.model)  # "Qwen/Qwen2.5-1.5B-Instruct"
```

**Why**: Prevents bugs from typos, centralizes configuration

---

#### `src/logger.py`
**Purpose**: Structured logging with GPU monitoring
**Lines**: 145
**Features**:
- Structured JSON logs (timestamp, level, event, context)
- GPU monitoring via `pynvml` (utilization %, memory usage)
- `LoggerContext` class for automatic timing

**Example**:
```python
from src.logger import get_logger, LoggerContext

logger = get_logger(__name__)

with LoggerContext(logger, "answer_question", question_id="q123"):
    # Your code here - automatically logs start, end, duration, GPU stats
    answer = agent.run(question)
```

**Output**:
```json
{
  "operation": "answer_question",
  "duration_ms": 11200,
  "question_id": "q123",
  "gpu": {"gpu_util_percent": 61, "gpu_memory_used_mb": 14070},
  "event": "answer_question_completed",
  "timestamp": "2026-04-20T23:19:16.254Z"
}
```

**Why**: Production observability - critical for debugging and monitoring

---

#### `src/data_loader.py`
**Purpose**: Load and preprocess FinQA dataset
**Lines**: 339
**Functions**:
- `download_finqa_file()` - Downloads from GitHub
- `load_finqa_dataset()` - Loads train/dev/test splits
- `prepare_example_for_rag()` - Formats for retrieval
  - Flattens nested JSON
  - Converts tables to Markdown
  - Combines pre_text + table + post_text

**Dataset**:
- **Train**: 6,251 examples
- **Dev**: 883 examples
- **Test**: 1,147 examples

**Example Output**:
```python
{
  "question": "what is the interest expense in 2009?",
  "answer": "45.0",
  "program": "divide(45, 1000)",
  "context": "Revenue increased 11%... [table]... Interest expense was $45M...",
  "table_str": "| Year | Revenue | ... |",
  "raw_table": [...original table...]
}
```

**Why**: Clean data loading with automatic downloads and caching

---

#### `src/retriever.py`
**Purpose**: Hybrid FAISS + BM25 retrieval
**Lines**: 475
**Class**: `FinQARetriever`

**Features**:
1. **Dense Retrieval (FAISS)**:
   - Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
   - Cosine similarity search
   - Fast vector indexing (6,251 docs in 275ms)

2. **Sparse Retrieval (BM25)**:
   - Keyword matching
   - Term frequency scoring
   - Complements semantic search

3. **Hybrid Fusion**:
   - Weighted combination: 70% FAISS + 30% BM25
   - Returns top-k documents with scores

**Methods**:
- `build_index()` - Embeds all docs, builds FAISS + BM25 indices
- `save_index()` - Saves to disk (`data/faiss_index/`)
- `load_index()` - Loads pre-built index
- `retrieve()` - FAISS-only retrieval
- `retrieve_hybrid()` - FAISS + BM25 fusion

**Usage**:
```python
retriever = FinQARetriever()
retriever.load_index()

results = retriever.retrieve_hybrid(
    query="what is the interest expense in 2009?",
    k=4,
    faiss_weight=0.7,
    bm25_weight=0.3
)
# Returns 4 docs with hybrid_score, context, question, answer
```

**Why**: Hybrid retrieval combines best of both worlds (semantic + keyword)

---

#### `src/agent.py` ⭐ **CORE FILE**
**Purpose**: LangGraph workflow with 5 nodes + retry mechanism
**Lines**: 733
**Architecture**:

```
┌─────────────┐
│  retrieve   │  → Get 4 relevant docs (FAISS + BM25)
└──────┬──────┘
       │
┌──────▼──────┐
│   reason    │  → LLM generates reasoning + calculation
└──────┬──────┘
       │
┌──────▼──────┐
│ calculator  │  → Evaluates arithmetic with sympy
└──────┬──────┘
       │
┌──────▼──────┐
│  verifier   │  → LLM fact-checks reasoning
└──────┬──────┘
       │
       ├─── PASS → answer (confident)
       ├─── UNCERTAIN → answer (human review)
       └─── FAIL → reason (retry with feedback)
                     ↑
                     └─── (up to 2 retries)
```

**Key Features**:

1. **State Management** (`AgentState` TypedDict):
   - Tracks question, docs, reasoning, calculations, verification
   - `retry_count`, `failure_feedback`, `retry_exhausted`
   - `trace` - full audit trail
   - `node_traces` - timing data per node

2. **Nodes**:
   - `retrieve_node` - Hybrid retrieval
   - `reason_node` - LLM reasoning (with retry support)
   - `calculator_node` - Safe arithmetic evaluation
   - `verifier_node` - Fact-checking
   - `answer_node` - Final answer formatting

3. **Conditional Routing** (`_route_after_verifier`):
   ```python
   if status == "PASS" and confidence == "HIGH":
       return "answer"  # CONFIDENT mode
   elif status == "FAIL" and retry_count < MAX_RETRIES:
       return "reason"  # RETRY with feedback
   elif retry_exhausted:
       return "answer"  # LOW_CONFIDENCE mode
   ```

4. **Self-Correction**:
   - Verifier detects issues (wrong numbers, bad arithmetic)
   - Injects issues into `failure_feedback`
   - Reason node retries with feedback in prompt
   - Up to 2 retries (3 total attempts)

5. **Observability**:
   - Every node logs start/end with `LoggerContext`
   - GPU metrics logged at each step
   - Node traces show timing breakdown

**Usage**:
```python
from src.agent import FinQAAgent

agent = FinQAAgent()
agent.load_index()

result = agent.run("what is the interest expense in 2009?")

print(result["final_answer"])
print(result["verification_status"])  # PASS/FAIL/UNCERTAIN
print(result["retry_count"])  # 0-2
```

**Why**: This is the heart of the system - production-ready agentic workflow

---

#### `src/vllm_server.py`
**Purpose**: Start vLLM server with GPU support
**Lines**: 354
**Features**:
- Auto-detects CUDA/MPS/CPU
- Configures tensor parallelism for multi-GPU
- GPU memory utilization settings
- Logs server startup status

**Usage**:
```bash
# Start with defaults from .env
python -m src.vllm_server

# Custom model
python -m src.vllm_server --model Qwen/Qwen2.5-7B-Instruct

# Custom GPU settings
python -m src.vllm_server --gpu-memory-utilization 0.8
```

**Note**: You ran vLLM directly in Colab, so this wasn't used

**Why**: Convenient wrapper for vLLM deployment

---

### **🧪 Testing Files**

#### `test_real_vllm.py`
**Purpose**: Test agent on specific validation questions
**Lines**: 200+
**Features**:
- Loads validation set
- Finds questions by text match
- Runs agent with real vLLM
- Prints full node trace, timeline, comparison

**Usage**:
```python
python test_real_vllm.py
```

**Output**: Detailed trace for each question (like what you saw in Colab)

**Why**: Validates end-to-end functionality with real LLM

---

### **📊 Data Files** (in `data/` directory)

#### `data/faiss_index/`
**Contains**:
- `faiss.index` - FAISS vector index (9.6 MB)
- `bm25.index` - BM25 keyword index (17.6 MB)
- `metadata.pkl` - Document metadata + tokenized contexts (59.7 MB)

**Total**: 87 MB (6,251 indexed documents)

**Why**: Pre-built index saves 5+ minutes on every startup

---

#### `data/raw/`
**Contains**:
- `train.json` - 6,251 training examples (74.6 MB)
- `dev.json` - 883 validation examples (10.4 MB)
- `test.json` - (not downloaded yet)

**Why**: Local cache of FinQA dataset (no re-downloading)

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| **Total Python Code** | ~1,900 lines |
| **Documentation** | ~1,000 lines |
| **Core Modules** | 7 files |
| **Dependencies** | 39 packages |
| **Indexed Documents** | 6,251 |
| **GPU Inference Time** | 7.2 seconds |
| **Retrieval Time** | 275ms |
| **Lines of Code (src/)** |  |
| - `agent.py` | 733 lines |
| - `retriever.py` | 475 lines |
| - `data_loader.py` | 339 lines |
| - `logger.py` | 145 lines |
| - `config.py` | 121 lines |
| - `vllm_server.py` | 354 lines |

---

## 🎯 What Makes This Production-Ready

1. ✅ **Type Safety**: Pydantic for config, TypedDict for state
2. ✅ **Observability**: Structured logs, GPU metrics, timing traces
3. ✅ **Error Handling**: Graceful degradation, retry mechanism
4. ✅ **Modularity**: Separate concerns (data, retrieval, agent, logging)
5. ✅ **Testability**: Test files, mock LLM support
6. ✅ **Documentation**: 5 MD files explaining everything
7. ✅ **Reproducibility**: requirements.txt, .env.example, Colab guide
8. ✅ **Self-Correction**: Verification + retry loop
9. ✅ **Performance**: Hybrid retrieval, GPU acceleration

---

## 🚀 For Runara Interview

**What Sets You Apart**:
1. **Real GPU deployment** (not just mock tests)
2. **Self-correcting agent** (retry with verification feedback)
3. **Production observability** (GPU metrics, structured logs)
4. **Hybrid retrieval** (FAISS + BM25 > either alone)
5. **Complete documentation** (RESULTS.md shows real metrics)

**Key Files to Highlight**:
- `src/agent.py` - Shows LangGraph + retry expertise
- `src/retriever.py` - Shows RAG implementation
- `RESULTS.md` - Shows real GPU performance
- `DEPLOYMENT_STRATEGY.md` - Shows production thinking

---

## 📌 Summary

**Essential Files** (18 total):
- **7 Python modules** - Core functionality
- **6 Documentation files** - Context and guides
- **3 Config files** - Environment setup
- **1 Test file** - Validation
- **1 Requirements file** - Dependencies

**Removed** (6 files):
- Mac-specific files (Ollama, SETUP_MAC.md)
- Temporary scripts (fix_context_length.py)
- Development tests (test_agent_mock.py)

**Total Project Size**: ~2.3M lines (mostly data files)
**Core Code**: ~2,200 lines (Python + docs)

---

**This is a complete, interview-ready FinQA agent!** 🎉
