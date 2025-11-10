#!/bin/bash
# Launch vLLM server for synthetic data generation (v3)
# Run this in a separate tmux session before running gen_syn_data_v3.py

MODEL_PATH=${1:-"audarai/auralix_flash_3"}
PORT=${2:-8000}
GPU_MEMORY=${3:-0.9}
TENSOR_PARALLEL=${4:-1}

echo "==============================================="
echo "Starting vLLM Server for Synthetic Data Gen"
echo "==============================================="
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "GPU Memory Utilization: $GPU_MEMORY"
echo "Tensor Parallel Size: $TENSOR_PARALLEL"
echo "==============================================="
echo ""
echo "Server will be available at: http://localhost:$PORT/v1"
echo ""

vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --trust-remote-code \
    --dtype float16 \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --disable-log-requests false \
    --log-level debug
