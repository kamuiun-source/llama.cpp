#pragma once

#include "common.cuh"

#include <string_view>

enum ggml_cuda_hc_projection {
    GGML_CUDA_HC_DOWN,
    GGML_CUDA_HC_UP,
    GGML_CUDA_HC_INJECT,
    GGML_CUDA_HC_PROJECTION_COUNT,
};

inline ggml_cuda_hc_projection ggml_cuda_hc_projection_for_weight(std::string_view name) {
    const auto ends_with = [&](std::string_view suffix) {
        return name.size() >= suffix.size() && name.substr(name.size() - suffix.size()) == suffix;
    };
    if (ends_with(".hc_attn_down.weight") || ends_with(".hc_ffn_down.weight")) {
        return GGML_CUDA_HC_DOWN;
    }
    if (ends_with(".hc_attn_up.weight") || ends_with(".hc_ffn_up.weight")) {
        return GGML_CUDA_HC_UP;
    }
    if (ends_with(".hc_attn_inject.weight") || ends_with(".hc_ffn_inject.weight")) {
        return GGML_CUDA_HC_INJECT;
    }
    return GGML_CUDA_HC_PROJECTION_COUNT;
}

int ggml_cuda_hc_mix_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
void ggml_cuda_hc_scale_silu(ggml_backend_cuda_context & ctx, const ggml_tensor * scale, ggml_tensor * dst);
bool ggml_cuda_hc_inject_split(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
