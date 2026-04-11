#!/usr/bin/env bash
# Start vLLM server for GLM-OCR with MTP speculative decoding
# Uses 70% GPU memory (~34.5 GiB) to leave room for layout detector (~4 GiB)
# Benchmarked: 0.70 matches 0.82 OCR speed with 3x more VRAM headroom
set -euo pipefail

vllm serve zai-org/GLM-OCR \
    --port 8080 \
    --served-model-name glm-ocr \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
    --allowed-local-media-path / \
    --gpu-memory-utilization 0.70
