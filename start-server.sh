#!/usr/bin/env bash
# llama-server launcher with GGSD prompt-cache-ssd enabled
set -euo pipefail

MODEL="/models/qwen3.8/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf"
DRAFT="/models/qwen3.8/mtp-Qwen3.8-27B-Q4_0.gguf"
HOST="0.0.0.0"
PORT=8080
SAVE_DIR="$(dirname "$0")/saves"
BIN="$(dirname "$0")/build-rocm/bin/llama-server"

if [ ! -f "$MODEL" ]; then
    echo "error: model not found: $MODEL" >&2
    exit 1
fi

if [ ! -f "$DRAFT" ]; then
    echo "error: MTP model not found: $DRAFT" >&2
    exit 1
fi

if [ ! -x "$BIN" ]; then
    echo "error: llama-server not found: $BIN (build it first: ./build_rocm.sh)" >&2
    exit 1
fi

mkdir -p "$SAVE_DIR"

# --prompt-cache-ssd: GGSD autoload + autosave (thresholds tunable:
#   --prompt-cache-ssd-min-prefix N  default 1024
#   --prompt-cache-ssd-margin N      default 256)
exec "$BIN" \
    -m "$MODEL" \
    -md "$DRAFT" \
    --n-gpu-layers all \
    --host "$HOST" \
    --port "$PORT" \
    --slot-save-path "$SAVE_DIR" \
    --prompt-cache-ssd \
    -np 4 \
    --kv-unified \
    --ctx-size 131072 \
    --cache-ram 8192 \
    --flash-attn on \
    --load-mode dio \
    --ctx-checkpoints 8 \
    --cont-batching \
    --spec-type ngram-map-k4v,draft-mtp \
    --spec-draft-type-k q8_0 \
    --spec-draft-type-v q8_0 \
    --spec-draft-n-max 3 \
    --spec-draft-threads 4 \
    --spec-draft-threads-batch 4 \
    --spec-ngram-map-k4v-size-n 32 \
    --spec-ngram-map-k4v-size-m 48 \
    --spec-ngram-map-k4v-min-hits 1 \
    "$@"
