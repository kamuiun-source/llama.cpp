#!/usr/bin/env bash
set -euo pipefail
src=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(cd -- "$src/../../.." && pwd)
out=${1:?usage: build.sh /path/to/output-directory}
mkdir -p -- "$out"
out=$(cd -- "$out" && pwd)
cuda_root=${CUDA_HOME:-/usr/local/cuda-12.9}
hipcxx=${HIPCXX:-/opt/rocm/bin/hipcc}
includes=(-I"$repo/ggml/include" -I"$repo/ggml/src" -I"$repo/ggml/src/ggml-cuda")
quantize="$repo/ggml/src/ggml-cuda/quantize.cu"
cuda_math=(-use_fast_math)
if [[ ${CUDA_STRICT:-0} == 1 ]]; then
    cuda_math=()
fi
printf '{ global: rtx_input_q8; local: *; };\n' > "$out/rtx.exports"
printf '{ global: mi50_input_*; local: *; };\n' > "$out/mi50.exports"
"$cuda_root/bin/nvcc" -shared -cudart shared -arch=sm_89 -std=c++17 -O3 "${cuda_math[@]}" \
    -Xcompiler=-fPIC,-ffunction-sections,-fdata-sections -Xlinker=--gc-sections -Xlinker="--version-script=$out/rtx.exports" \
    "${includes[@]}" "$src/rtx.cu" "$quantize" -o "$out/libbench-rtx-q8.so"
"$hipcxx" -shared -fPIC -x hip --offload-arch=gfx906 -std=c++17 -O3 -DGGML_USE_HIP \
    -ffunction-sections -fdata-sections -Wl,--gc-sections -Wl,"--version-script=$out/mi50.exports" \
    "${includes[@]}" "$src/mi50.cpp" "$quantize" -o "$out/libbench-mi50-q8.so"
"${CXX:-c++}" -std=c++17 -O3 -Wall -Wextra -I"$cuda_root/include" "$src/bench.cpp" \
    -L"$cuda_root/lib64/stubs" -L"$out" -Wl,-rpath,'$ORIGIN' -lcuda -lbench-rtx-q8 -lbench-mi50-q8 -o "$out/rtx-mi50-input-q8"
