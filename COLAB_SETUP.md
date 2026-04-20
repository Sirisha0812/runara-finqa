# Google Colab Setup Guide

## Prerequisites
1. Create a GitHub Personal Access Token (if private repo):
   - Go to: https://github.com/settings/tokens/new
   - Note: `runara-finqa-colab`
   - Expiration: 7 days
   - Scopes: ✅ `repo` (Full control of private repositories)
   - Copy the token (e.g., `ghp_abc123...`)

---

## Quick Start in Colab

### Option A: Public Repo (Simple)
```python
# Clone repo
!git clone https://github.com/Sirisha0812/runara-finqa.git
%cd runara-finqa
```

### Option B: Private Repo (With Token)
```python
# Clone with token (replace YOUR_TOKEN)
!git clone https://YOUR_TOKEN@github.com/Sirisha0812/runara-finqa.git
%cd runara-finqa
```

---

## Complete Setup Script

Copy this into a **single Colab cell**:

```python
# ========================================
# FinQA Agent Setup on Google Colab
# ========================================

# 1. Clone repository
!git clone https://github.com/Sirisha0812/runara-finqa.git
%cd runara-finqa

# 2. Install dependencies
print("\n📦 Installing dependencies...")
!pip install -q vllm>=0.6.0
!pip install -q -r requirements.txt

# 3. Check GPU
import torch
print(f"\n🔧 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"   CUDA available: {torch.cuda.is_available()}")

# 4. Start vLLM server in background
print("\n🚀 Starting vLLM server...")
!nohup vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --port 8000 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    > vllm_server.log 2>&1 &

# 5. Wait for server to start
import time
print("   Waiting for vLLM server to start (30 seconds)...")
time.sleep(30)

# 6. Check if server is running
print("\n✅ Checking server status...")
!curl -s http://localhost:8000/v1/models | head -20

# 7. Update .env to point to local server
print("\n⚙️  Updating configuration...")
!echo "VLLM_API_BASE=http://localhost:8000/v1" > .env
!echo "VLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct" >> .env
!cat .env.example | grep -v VLLM_API_BASE | grep -v VLLM_MODEL >> .env

print("\n" + "="*60)
print("✅ SETUP COMPLETE!")
print("="*60)
print("\nServer running at: http://localhost:8000/v1")
print("Model: Qwen/Qwen2.5-1.5B-Instruct")
print("\nTo run the agent:")
print("  !python -m src.agent")
print("\nTo check server logs:")
print("  !tail -20 vllm_server.log")
print("="*60)
```

---

## Test the Agent

After setup, run:

```python
# Test on demo question
!python -m src.agent
```

Or test on specific questions:

```python
!python test_real_vllm.py
```

---

## Troubleshooting

### Check vLLM server logs
```python
!tail -50 vllm_server.log
```

### Restart vLLM server
```python
# Kill existing server
!pkill -f vllm

# Start new server
!nohup vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --port 8000 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    > vllm_server.log 2>&1 &

# Wait 30 seconds
import time
time.sleep(30)
```

### Check GPU memory
```python
!nvidia-smi
```

### Test vLLM endpoint directly
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=50
)

print(response.choices[0].message.content)
```

---

## Expected Output

When you run `!python -m src.agent`, you should see:

```
================================================================================
FinQA Agent — LangGraph Workflow Demo
================================================================================

Loading hybrid retriever index...
✅ Hybrid FAISS + BM25 index loaded successfully.

Test Query: what is the interest expense in 2009?

[NODE: RETRIEVE]
  Documents retrieved: 4
  Top hybrid score: 0.7505

[NODE: REASON]
  Reasoning: ...
  Calculation: ...

[NODE: CALCULATOR]
  Result: ...

[NODE: VERIFIER]
  Status: PASS
  Confidence: HIGH

[NODE: ANSWER]
  Final Answer: ...

TOTAL TIME: 5.2 seconds
```

---

## Next Steps

1. ✅ Run the agent on demo questions
2. ✅ Test on your own FinQA questions
3. 📊 Evaluate on validation set:
   ```python
   !python -m src.evaluate --num-questions 100
   ```
4. 📝 Review results and metrics

---

## Notes

- **GPU Required**: T4 or better (12GB+ VRAM)
- **Model Size**: Qwen2.5-1.5B (~3GB download)
- **First run**: Takes ~5 minutes to download model
- **Subsequent runs**: Instant (model cached)

---

## Files in This Repo

```
runara-finqa/
├── src/
│   ├── agent.py              # LangGraph workflow
│   ├── retriever.py          # Hybrid FAISS + BM25
│   ├── data_loader.py        # FinQA dataset loader
│   ├── config.py             # Configuration
│   ├── logger.py             # Structured logging
│   ├── vllm_server.py        # vLLM server launcher
│   └── ollama_server.py      # Mac alternative
├── data/
│   ├── faiss_index/          # Pre-built index (6251 docs)
│   └── raw/                  # FinQA train/dev data
├── test_agent_mock.py        # Test with mock LLM
├── test_real_vllm.py         # Test with real vLLM
├── requirements.txt          # Dependencies
└── README.md                 # Project overview
```
