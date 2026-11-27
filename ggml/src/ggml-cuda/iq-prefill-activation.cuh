#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ggml_hip_narrow_iq_prefill {

// Same logical row-major stride as MMVQ, with the D4 scale instead of half scale and sum.
struct block_q8_1_d4 {
    float d;
    int8_t qs[QK8_1];
};
static_assert(sizeof(block_q8_1_d4) == sizeof(block_q8_1));
static_assert(alignof(block_q8_1_d4) == alignof(block_q8_1));
static_assert(offsetof(block_q8_1_d4, qs) == offsetof(block_q8_1, qs));

static __device__ __forceinline__ float activation_scale(const block_q8_1 & block) {
    return __low2float(block.ds);
}

static __device__ __forceinline__ float activation_scale(const block_q8_1_d4 & block) {
    return block.d;
}

enum class mode : unsigned { legacy, fraction, d4, fraction_d4 };

__host__ __device__ constexpr bool preserve_fraction(mode m) {
    return (unsigned(m) & unsigned(mode::fraction)) != 0;
}

__host__ __device__ constexpr bool d4_activation(mode m) {
    return (unsigned(m) & unsigned(mode::d4)) != 0;
}

template<mode m>
using activation_t = std::conditional_t<d4_activation(m), block_q8_1_d4, block_q8_1>;

template<typename Launch>
void launch(mode m, const Launch & run) {
    switch (m) {
        case mode::legacy:      run(std::integral_constant<mode, mode::legacy>{}); break;
        case mode::fraction:    run(std::integral_constant<mode, mode::fraction>{}); break;
        case mode::d4:          run(std::integral_constant<mode, mode::d4>{}); break;
        case mode::fraction_d4: run(std::integral_constant<mode, mode::fraction_d4>{}); break;
    }
}

} // namespace ggml_hip_narrow_iq_prefill
