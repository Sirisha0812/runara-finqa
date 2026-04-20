# Mac Setup Guide (Apple Silicon)

Since you're on a MacBook Air (Apple Silicon), vLLM won't work. Use **Ollama** instead - it's optimized for Apple Silicon with MPS acceleration.

---

## Quick Start

### 1. Install Ollama

```bash
# Option 1: Download from website
open https://ollama.com/download

# Option 2: Use Homebrew
brew install ollama
```

### 2. Update .env to use Ollama

```bash
# Backup current .env
cp .env .env.vllm_backup

# Use Ollama configuration
cp .env.ollama .env
```

Your `.env` should now have:
```bash
VLLM_API_BASE=http://localhost:11434/v1
VLLM_MODEL=qwen2.5:32b
```

### 3. Start Ollama Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Start Ollama server (auto-pulls model if needed)
python -m src.ollama_server
```

**Expected output:**
```
✅ OLLAMA SERVER READY
================================================================================

Server running at: http://localhost:11434
Model: qwen2.5:32b
Device: Apple Silicon (MPS)

Ollama API is OpenAI-compatible at:
  http://localhost:11434/v1
```

### 4. Run the Agent

Open a **new terminal** (keep Ollama running in the first):

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the agent
python -m src.agent
```

---

## Model Recommendations for Mac

| Model | Size | RAM Required | Speed | Quality |
|-------|------|--------------|-------|---------|
| `qwen2.5:7b` | 7B | 8GB | Fast | Good |
| `qwen2.5:14b` | 14B | 12GB | Medium | Better |
| `qwen2.5:32b` | 32B | 20GB | Slow | Best |
| `deepseek-r1:32b` | 32B | 20GB | Slow | Excellent reasoning |

**For MacBook Air (8GB RAM):** Use `qwen2.5:7b`
**For MacBook Air (16GB RAM):** Use `qwen2.5:14b` or `qwen2.5:32b`

To change models:
```bash
# Start with smaller model
python -m src.ollama_server --model qwen2.5:7b

# Update .env
VLLM_MODEL=qwen2.5:7b
```

---

## Why Ollama Instead of vLLM?

| Feature | vLLM | Ollama |
|---------|------|--------|
| **CUDA Support** | ✅ Yes | ❌ No |
| **Apple Silicon (MPS)** | ❌ No | ✅ Yes |
| **Mac CPU** | ❌ No | ✅ Yes |
| **OpenAI API Compatible** | ✅ Yes | ✅ Yes |
| **Performance on Mac** | ❌ Won't run | ✅ Excellent |

**vLLM requires NVIDIA GPUs** - it won't run on Mac at all.

**Ollama is built for Mac** - uses Apple's Metal Performance Shaders (MPS) for GPU acceleration.

---

## Troubleshooting

### Ollama server won't start
```bash
# Check if already running
curl http://localhost:11434/api/tags

# Kill existing instance
pkill ollama

# Start manually
ollama serve
```

### Model download is slow
Ollama downloads models from their CDN. Large models (32B) can take 10-20 minutes on slow connections.

```bash
# Check download progress
ollama list
```

### Out of memory
If the model is too large for your Mac:
```bash
# Use smaller model
python -m src.ollama_server --model qwen2.5:7b
```

---

## Testing

### 1. Test Ollama API directly
```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:32b",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

### 2. Test with agent
```bash
python -m src.agent
```

### 3. Test with mock LLM (no Ollama needed)
```bash
python test_agent_mock.py
```

---

## For Production (GPU Server)

If you deploy to a GPU server with NVIDIA GPUs, switch back to vLLM:

```bash
# Restore vLLM config
cp .env.vllm_backup .env

# Start vLLM server (on GPU machine)
python -m src.vllm_server
```

vLLM is **much faster** than Ollama on NVIDIA GPUs (10-20x speedup).

---

## Next Steps

1. ✅ Start Ollama: `python -m src.ollama_server`
2. ✅ Run agent: `python -m src.agent`
3. ✅ Test on 3 questions: `python test_agent_mock.py`
4. 📝 Write README.md
5. 🎯 Evaluate on full validation set
