#!/usr/bin/env bash
# llama-server launcher with GGSD prompt-cache-ssd enabled
set -euo pipefail

MODEL="/models/unsloth/qwen3.5-gguf/Qwen3.5-9B-UD-Q4_K_XL.gguf"
HOST="0.0.0.0"
PORT=8080
SAVE_DIR="$(dirname "$0")/saves"
BIN="$(dirname "$0")/build-linux/bin/llama-server"

if [ ! -f "$MODEL" ]; then
    echo "error: model not found: $MODEL" >&2
    exit 1
fi

if [ ! -x "$BIN" ]; then
    echo "error: llama-server not found: $BIN (build it first: cmake --build build-linux --target llama-server)" >&2
    exit 1
fi

mkdir -p "$SAVE_DIR"

# --prompt-cache-ssd: GGSD autoload + autosave (thresholds tunable:
#   --prompt-cache-ssd-min-prefix N  default 1024
#   --prompt-cache-ssd-margin N      default 256)
exec "$BIN" \
    -m "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --slot-save-path "$SAVE_DIR" \
    --prompt-cache-ssd \
    -np 4 \
    --kv-unified \
    --cache-ram 8192 \
    --flash-attn auto \
    --load-mode dio \
    --ctx-checkpoints 8 \
    --cont-batching \
    "$@"
