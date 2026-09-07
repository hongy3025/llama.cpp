#!/usr/bin/env bash
# ROCmFPX dense-Qwen MTP decode-fast profile for Strix Halo / ROCm0.
set -euo pipefail

MODEL="${MODEL:-/models/qwen3.8/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf}"
DRAFT="${DRAFT:-/models/qwen3.8/mtp-Qwen3.8-27B-Q4_0.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
BIN="$(dirname "$0")/build-rocm/bin/llama-server"

# Match q38rocm's supported Strix Halo HIP runtime profile without changing
# machine-wide DPM, THP, or GTT settings.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"
export GGML_HIP_ENABLE_UNIFIED_MEMORY="${GGML_HIP_ENABLE_UNIFIED_MEMORY:-1}"
export ROCM_FLUSH_ACCEPT="${ROCM_FLUSH_ACCEPT:-1}"

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

# q38rocm dense-Qwen MTP profile with asymmetric TurboQuant KV:
# target K=q8_0, V=turbo4; independent draft K=q8_0, V=turbo4.
# TurboQuant is a runtime KV-cache format, not a model-weight format.
# Boundary protection can be enabled externally with:
# LLAMA_KV_TURBO_BOUNDARY_LAYERS=2 LLAMA_KV_TURBO_BOUNDARY_V=1
exec "$BIN" \
    -m "$MODEL" \
    -md "$DRAFT" \
    -dev ROCm0 \
    --spec-draft-device ROCm0 \
    --n-gpu-layers all \
    --spec-draft-ngl all \
    --host "$HOST" \
    --port "$PORT" \
    -np 1 \
    --ctx-size 262144 \
    -b 512 \
    -ub 512 \
    -t 16 \
    -tb 32 \
    --poll 100 \
    --flash-attn on \
    --mmap \
    --cache-type-k q8_0 \
    --cache-type-v turbo4 \
    --cache-ram 8192 \
    --ctx-checkpoints 0 \
    --jinja \
    --reasoning off \
    --reasoning-format deepseek \
    --reasoning-budget -1 \
    --no-context-shift \
    --no-mmproj \
    --temp 0 \
    --top-p 0.95 \
    --top-k 20 \
    --seed 123 \
    --spec-type draft-mtp \
    --spec-draft-type-k q8_0 \
    --spec-draft-type-v turbo4 \
    --spec-draft-n-max 4 \
    --spec-draft-n-min 0 \
    --spec-draft-p-min 0.75 \
    --spec-draft-p-split 0.10 \
    --spec-draft-threads 16 \
    --spec-draft-threads-batch 32 \
    --no-spec-draft-backend-sampling \
    --spec-draft-poll 1 \
    --spec-draft-poll-batch 1 \
    --metrics \
    "$@"
