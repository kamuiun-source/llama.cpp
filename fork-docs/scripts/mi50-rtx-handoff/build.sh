#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
rocm_root=${ROCM_PATH:-/opt/rocm}
cuda_root=${CUDA_HOME:-/usr/local/cuda}
cxx=${CXX:-g++}
"${HIPCXX:-$rocm_root/bin/hipcc}" --offload-arch=gfx906 -std=c++17 -O2 -c hip-source.cpp -o hip-source.o
"$cxx" -std=c++17 -O2 -Wall -Wextra -I"$cuda_root/include" -c handoff.cpp -o handoff.o
"$cxx" -no-pie handoff.o hip-source.o -L"$cuda_root/lib64/stubs" -L"$rocm_root/lib" \
    -Wl,-rpath,"$rocm_root/lib" -lcuda -lamdhip64 -o handoff
