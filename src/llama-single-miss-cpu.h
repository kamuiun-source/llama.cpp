#pragma once

#include "ggml-cpu.h"

#include <array>
#include <cmath>
#include <cstdlib>

inline bool llama_single_miss_cpu_configured() {
    const char * path = std::getenv("LLAMA_EXPERT_HIP_MTP3_SINGLE_MISS_CPU_MODE_FILE");
    return path && *path;
}

struct llama_single_miss_cpu {
    static constexpr int width = 2560, inner = 640, tokens = 3, ranks = 10;
    std::array<float, width * tokens> weighted;
    size_t eligible = 0, applied = 0, positions = 0, avoided_bytes = 0;
    int64_t compute_us = 0;

    bool compute(const std::array<const ggml_tensor *, 3> & planes, int host_slot, int expert,
            const int32_t * ids, const float * activation, const float * router_weights, unsigned & mask) {
        using get_traits_t = const ggml_type_traits_cpu * (*)(ggml_type);
        static const auto get_traits = []() -> get_traits_t {
            const auto reg = ggml_backend_reg_by_name("CPU");
            return reg ? reinterpret_cast<get_traits_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_get_single_miss_traits")) : nullptr;
        }();
        if (!get_traits || host_slot < 0) {
            return false;
        }
        const ggml_type_traits_cpu * traits[3];
        const ggml_type_traits_cpu * input_traits[3];
        for (int p = 0; p < 3; ++p) {
            const auto * w = planes[p];
            if (!w || w->ne[0] != (p == 2 ? inner : width) || w->ne[1] != (p == 2 ? width : inner) ||
                    host_slot >= w->ne[2] || !w->data || !w->buffer || !ggml_backend_buffer_is_host(w->buffer)) {
                return false;
            }
            traits[p] = get_traits(w->type);
            if (!traits[p] || !traits[p]->vec_dot) {
                return false;
            }
            input_traits[p] = get_traits(traits[p]->vec_dot_type);
            if (!input_traits[p] || !input_traits[p]->from_float || ggml_row_size(traits[p]->vec_dot_type, w->ne[0]) > 4096) {
                return false;
            }
        }

        const auto run = [&](int plane, const float * input, float * output) {
            const auto * w = planes[plane];
            const auto * data = static_cast<const char *>(w->data) + host_slot * w->nb[2];
            alignas(64) char quantized[4096];
            input_traits[plane]->from_float(input, quantized, w->ne[0]);
            for (int r = 0; r < w->ne[1]; ++r) {
                traits[plane]->vec_dot(w->ne[0], output + r, 0, data + r*w->nb[1], 0, quantized, 0, 1);
            }
        };

        mask = 0;
        for (int t = 0; t < tokens; ++t) {
            for (int r = 0; r < ranks; ++r) {
                if (ids[t*ranks + r] != expert) {
                    continue;
                }
                if (mask & (1u << t)) {
                    return false;
                }
                std::array<float, inner> gate, up;
                run(0, activation + t*width, gate.data());
                run(1, activation + t*width, up.data());
                for (int j = 0; j < inner; ++j) {
                    gate[j] = (gate[j] / (1.0f + std::exp(-gate[j]))) * up[j];
                }
                auto * out = weighted.data() + t*width;
                run(2, gate.data(), out);
                for (int j = 0; j < width; ++j) {
                    out[j] *= router_weights[t*ranks + r];
                }
                mask |= 1u << t;
            }
        }
        return mask != 0;
    }
};
