#pragma once

#include "common.cuh"

int ggml_cuda_hc_mix_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
