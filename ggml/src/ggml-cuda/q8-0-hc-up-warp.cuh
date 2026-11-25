#pragma once

#include "q8-0-hc-up-packed.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static constexpr int hc_up_warps_per_block = 4;

template <bool packed>
__launch_bounds__(hc_up_warps_per_block * ggml_cuda_get_physical_warp_size(), 1)
static __global__ void hc_up_q8_0_warp(
        const block_q8_0 * weights, hc_up_packed_weights packed_weights, const block_q8_1 * activations, float * dst,
        int stride_col_y, int stride_col_dst) {
    constexpr int width = ggml_cuda_get_physical_warp_size();
    constexpr int lanes_per_quant = QI8_0 / VDR_Q8_0_Q8_1_MMVQ;
    constexpr int blocks_per_row = 320 / QK8_0;
    constexpr int blocks_per_warp = width / lanes_per_quant;
    const int lane = threadIdx.x;
    const int first_block = lane / lanes_per_quant;
    const int tail_block = first_block + blocks_per_warp;
    const int iqs = VDR_Q8_0_Q8_1_MMVQ * (lane % lanes_per_quant);
    const int row = 2 * (hc_up_warps_per_block * blockIdx.x + threadIdx.y);
    ggml_cuda_pdl_sync();

#pragma unroll
    for (int col = 0; col < 3; ++col) {
        const auto * x = activations + col * stride_col_y;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int base = (row + r) * blocks_per_row;
            const auto dot = [&](int block) {
                if constexpr (packed) {
                    return hc_up_dot_packed(packed_weights, base + block, iqs, x[block]);
                }
                return vec_dot_q8_0_q8_1(weights + base, x + block, block, iqs);
            };
            float sum = dot(first_block);
            if (tail_block < blocks_per_row) {
                const float tail = dot(tail_block);
                // Match the rounded addition of the second warp's shared partial.
                sum = __fadd_rn(sum, tail);
            }
            sum = warp_reduce_sum<width>(sum);
            if (lane == r) {
                dst[col * stride_col_dst + row + r] = sum;
            }
        }
    }
}
#endif
