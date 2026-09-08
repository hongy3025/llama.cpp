#!/usr/bin/env bash
set -euo pipefail

HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    GGML_HIP_ENABLE_UNIFIED_MEMORY=1 \
    ROCM_FLUSH_ACCEPT=1 \
    exec "$(dirname "$0")/build-rocm/bin/llama-server" \
    --model /models/qwen3.8/julianmb__Qwen3.8-27B-ROCmFP4-STRIX_LEAN.gguf \
    --device ROCm0 \
    --gpu-layers all \
    --flash-attn on \
    --parallel 1 \
    --ctx-size 131072 \
    --batch-size 2048 \
    --ubatch-size 1024 \
    --threads 16 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --cont-batching \
    --kv-unified \
    --cache-prompt \
    --ctx-checkpoints 8 \
    --checkpoint-min-step 4096 \
    --cache-ram 32768 \
    --spec-type draft-mtp \
    --spec-draft-n-max 4 \
    --spec-draft-p-min 0.0 \
    --temperature 0.0 \
    --repeat-penalty 1.05 \
    --reasoning auto \
    --host 0.0.0.0 \
    --port 8080
