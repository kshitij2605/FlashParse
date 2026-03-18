#!/usr/bin/env bash
# Start vLLM server for GLM-OCR with MTP speculative decoding
set -euo pipefail

vllm serve zai-org/GLM-OCR \
    --port 8080 \
    --served-model-name glm-ocr \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
    --allowed-local-media-path /
