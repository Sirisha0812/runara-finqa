#!/bin/bash
# Start vLLM server with Qwen/Qwen2.5-32B-Instruct model
#
# Requirements:
# - GPU with at least 48GB VRAM (for 32B model)
# - Or use a smaller model like Qwen/Qwen2.5-7B-Instruct (16GB VRAM)
#
# Usage: ./start_vllm.sh

# Activate virtual environment
source .venv/bin/activate

# Start vLLM server
# Note: This will use GPU 0 by default
# Adjust --tensor-parallel-size based on your GPUs
vllm serve Qwen/Qwen2.5-32B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code

# Alternative: Use smaller model for Mac/CPU testing
# vllm serve Qwen/Qwen2.5-7B-Instruct \
#     --port 8000 \
#     --device cpu \
#     --max-model-len 2048
