#!/usr/bin/env bash
set -euo pipefail

# ROCm 10.0 (TheRock) environment - matches Dockerfile.rocm-10.0-performance.
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=/opt/rocm/bin:/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/core/lib/rocm_sysdeps/lib:/opt/rocm/core/lib

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Install build dependencies if not already present.
if ! rpm -q amdrocm-core-devel10.0-gfx1151 &>/dev/null; then
    echo "Installing ROCm 10.0 build dependencies..."
    dnf install -y --nodocs --setopt=install_weak_deps=False \
        amdrocm-core-devel10.0-gfx1151
fi

# Clean previous build state for a fresh configure.
rm -rf build-rocm

cmake -S . -B build-rocm \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1151 \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_RPC=ON \
    -DROCM_PATH=/opt/rocm \
    -DHIP_PLATFORM=amd \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_TESTS_INSTALL=OFF

cmake --build build-rocm --config Release -- -j"$(nproc)"
