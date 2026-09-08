#!/usr/bin/env bash
set -euo pipefail

SAVE_DIR="$(dirname "$0")/saves"
mkdir -p "$SAVE_DIR"

HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    GGML_HIP_ENABLE_UNIFIED_MEMORY=1 \
    ROCM_FLUSH_ACCEPT=1 \
    exec "$(dirname "$0")/build-rocm/bin/llama-server" \
    --model /models/qwen3.8/julianmb__Qwen3.8-27B-ROCmFP4-STRIX_LEAN.gguf \
    --device ROCm0 \
    --gpu-layers all \
    --flash-attn on \
    --parallel 2 \
    --ctx-size 131072 \
    --batch-size 2048 \
    --ubatch-size 1024 \
    --threads 16 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --cont-batching \
    --kv-unified \
    --cache-prompt \
    --slot-save-path "$SAVE_DIR" \
    --prompt-cache-ssd \
    --prompt-cache-ssd-min-prefix 256 \
    --prompt-cache-ssd-margin 64 \
    --ctx-checkpoints 8 \
    --checkpoint-min-step 4096 \
    --cache-ram 32768 \
    --spec-type ngram-map-k4v,draft-mtp \
    --spec-draft-n-max 2 \
    --spec-draft-p-min 0.0 \
    --spec-ngram-map-k4v-size-n 32 \
    --spec-ngram-map-k4v-size-m 48 \
    --spec-ngram-map-k4v-min-hits 1 \
    --temperature 0.0 \
    --repeat-penalty 1.05 \
    --reasoning auto \
    --host 0.0.0.0 \
    --port 8080
