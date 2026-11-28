#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
out=${1:?usage: build.sh /path/to/output}
exec "${HIPCXX:-/opt/rocm/bin/hipcc}" -x hip --offload-arch=gfx906 -std=c++17 -O3 -DGGML_USE_HIP \
    -ffunction-sections -fdata-sections -Wl,--gc-sections \
    -I"$repo/ggml/include" -I"$repo/ggml/src" -I"$repo/ggml/src/ggml-cuda" \
    "$repo/fork-docs/scripts/mapped-input-q8/bench.cpp" "$repo/ggml/src/ggml-cuda/quantize.cu" -o "$out"
