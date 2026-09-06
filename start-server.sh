#!/usr/bin/env bash
# llama-server launcher with GGSD prompt-cache-ssd enabled
set -euo pipefail

MODEL="/models/LuffyTheFox/Hermes3.6-35B-A3B/Hermes3.6-35B-A3B-Uncensored-Genesis-V9-MTP-APEX.gguf"
HOST="0.0.0.0"
PORT=8080
SAVE_DIR="$(dirname "$0")/saves"
BIN="$(dirname "$0")/build-rocm/bin/llama-server"

if [ ! -f "$MODEL" ]; then
    echo "error: model not found: $MODEL" >&2
    exit 1
fi

if [ ! -x "$BIN" ]; then
    echo "error: llama-server not found: $BIN (build it first: cmake --build build-rocm --target llama-server)" >&2
    exit 1
fi

mkdir -p "$SAVE_DIR"

# --prompt-cache-ssd: GGSD autoload + autosave.
#   GGSD segments are 256 tokens; the thresholds below are tunable.
exec "$BIN" \
    -m "$MODEL" \
    --n-gpu-layers all \
    --mmproj /models/LuffyTheFox/Hermes3.6-35B-A3B/mmproj-Hermes3.6-35B-A3B-Uncensored-Genesis-F16.gguf \
    --host "$HOST" \
    --port "$PORT" \
    --slot-save-path "$SAVE_DIR" \
    --prompt-cache-ssd \
    --prompt-cache-ssd-min-prefix 256 \
    --prompt-cache-ssd-margin 64 \
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
