#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace ggml_gpu_output_modes {

inline constexpr char mode_file_env[] = "LLAMA_GPU_OUTPUT_MODE_FILE";
inline constexpr char setter_proc[] = "ggml_backend_set_gpu_output_modes";

enum flag : unsigned {
    iq_sign_decode,
    iq2_paired_dot,
    iq4_compact_decode,
    iq3_lds_prefetch,
    iq3_lds_row_groups,
    iq2_lds_prefetch,
    iq2_lds_row_groups,
    iq2_no_lds_row_groups,
    hc_down_split_k,
    gdn_gated_norm,
    concat_small,
    unary_mul_bcast,
    iq_mmvq_preserve_fraction,
    count,
};

inline constexpr std::array<std::string_view, count> names{{
    "GGML_HIP_GFX906_IQ_SIGN_DECODE",
    "GGML_HIP_GFX906_IQ2_S_PAIRED_DOT",
    "GGML_HIP_GFX906_IQ4_NL_DOWN_COMPACT_DECODE",
    "GGML_HIP_GFX906_IQ3_XXS_LDS_PREFETCH",
    "GGML_HIP_GFX906_IQ3_XXS_LDS_ROW_GROUPS",
    "GGML_HIP_GFX906_IQ2_S_LDS_PREFETCH",
    "GGML_HIP_GFX906_IQ2_S_LDS_ROW_GROUPS",
    "GGML_HIP_GFX906_IQ2_S_NO_LDS_ROW_GROUPS",
    "GGML_CUDA_HC_DOWN_SPLIT_K",
    "GGML_CUDA_GDN_GATED_NORM",
    "GGML_CUDA_CONCAT_SMALL",
    "GGML_CUDA_UNARY_MUL_BCAST",
    "GGML_HIP_GFX906_IQ_MMVQ_PRESERVE_FRACTION",
}};

constexpr uint32_t bit(unsigned index) {
    return uint32_t(1) << index;
}

constexpr uint32_t pack(uint32_t present, uint32_t enabled) {
    return (present << count) | enabled;
}

constexpr uint32_t common_bits = bit(concat_small) | bit(unary_mul_bcast);
constexpr uint32_t gfx906_bits = (bit(hc_down_split_k) - 1) | bit(iq_mmvq_preserve_fraction);
constexpr uint32_t ada_bits = bit(hc_down_split_k) | bit(gdn_gated_norm);
constexpr uint32_t hip_bits = common_bits | gfx906_bits;
constexpr uint32_t cuda_bits = common_bits | ada_bits;

} // namespace ggml_gpu_output_modes
