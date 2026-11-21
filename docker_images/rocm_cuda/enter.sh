#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
models_dir=${MODELS_DIR:-/run/media/t-saka/ボリューム/AI/share/llm_models}
mi50_render_path=${MI50_RENDER:-/dev/dri/by-path/pci-0000:03:00.0-render}

if [[ ! -d "$models_dir" ]]; then
    printf 'Model directory does not exist: %s\n' "$models_dir" >&2
    exit 1
fi
models_dir=$(cd -- "$models_dir" && pwd -P)

if [[ ! -c /dev/kfd ]]; then
    printf 'ROCm device /dev/kfd is missing or is not a character device.\n' >&2
    exit 1
fi
if ! mi50_render=$(readlink -e -- "$mi50_render_path") || [[ ! -c "$mi50_render" ]]; then
    printf 'MI50 render device is missing or is not a character device: %s\n' "$mi50_render_path" >&2
    exit 1
fi

kfd_gid=$(stat -c '%g' -- /dev/kfd)
render_gid=$(stat -c '%g' -- "$mi50_render")
gpu_groups=(--group-add "$kfd_gid")
if [[ "$render_gid" != "$kfd_gid" ]]; then
    gpu_groups+=(--group-add "$render_gid")
fi

exec sudo docker run --rm -it --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e HOME=/tmp -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 \
    --user "$(id -u):$(id -g)" "${gpu_groups[@]}" \
    --device=/dev/kfd --device="$mi50_render" \
    --volume "$repo_root:/workspace/llama.cpp" \
    --volume "$models_dir:/models:ro" \
    --workdir /workspace/llama.cpp \
    --entrypoint /bin/bash llama-cuda-rocm-dev
